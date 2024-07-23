# come up with conversations that occur in real life
from functools import partial
import pickle

import click
from dsp import Claude
import dspy
from pydantic import BaseModel, Field
import ray
from ray.util.actor_pool import ActorPool
from tqdm import tqdm



class SuggestActivities(dspy.Signature):
    """
    Please list common activities or locations that could involve people talking to each other.
    Don't include activities which have already been mentioned.
    Try to generate exactly `n` suggestions.
    """
    n: int = dspy.InputField(default=10)
    already_mentioned: list[str] = dspy.InputField(default=[])
    activities: list[str] = dspy.OutputField()


class SuggestActivityTopics(dspy.Signature):
    """
    Please list topics of ordinary, casual conversation that might occur when doing the activity.
    Don't include topics which have already been mentioned.
    Try to generate exactly `n` suggestions.
    """
    n: int = dspy.InputField(default=10)
    activity: str = dspy.InputField()
    already_mentioned: list[str] = dspy.InputField(default=[])
    topics: list[str] = dspy.OutputField()


class SuggestSpecificTopics(dspy.Signature):
    """
    Given the topic at hand, suggest a list of more specific subtopics.
    Don't include subtopics which have already been mentioned.
    The subtopic name should imply an emotional context if possible.
    Try to generate exactly `n` suggestions.
    """
    n: int = dspy.InputField(default=10)
    topic: str = dspy.InputField(desc="Topic at hand")
    already_mentioned: list[str] = dspy.InputField(default=[])
    subtopics: list[str] = dspy.OutputField()


def build_list(predictor, output_attr, desired_n, truncate=True):
    outputs = set()
    while len(outputs) < desired_n:
        new_outs = predictor(already_mentioned=list(outputs), n=desired_n - len(outputs))
        outputs |= set(getattr(new_outs, output_attr))
        # print(f'Generated {len(outputs)} / {desired_n}')

    outputs = list(outputs)

    if truncate:
        outputs = outputs[:desired_n]

    return outputs


@ray.remote
class PredictionAgent:
    def __init__(self):
        self.model = Claude(model="claude-3-5-sonnet-20240620", max_tokens=4096)
        dspy.settings.configure(lm=self.model)
        self.pred_activities = dspy.TypedPredictor(SuggestActivities)
        self.pred_topics = dspy.TypedPredictor(SuggestActivityTopics)
        self.pred_subtopics = dspy.TypedPredictor(SuggestSpecificTopics)

    def predict_activities(self, n_activities):
        return build_list(self.pred_activities, 'activities', n_activities)

    def predict_topics(self, n_topics, activity):
        topics = build_list(
            partial(self.pred_topics, activity=activity),
            'topics',
            n_topics
        )
        return topics

    def predict_subtopics(self, n_subtopics, full_topic):
        subtopics = build_list(
            partial(self.pred_subtopics, topic=full_topic),
            'subtopics',
            n_subtopics
        )
        return subtopics


@click.command()
@click.option('--output_file', default='topics.pkl')
@click.option('--n_activities', type=int, default=2)
@click.option('--n_topics', type=int, default=2)
@click.option('--n_subtopics', type=int, default=2)
@click.option('--n_workers', type=int, default=10)
def main(output_file, n_activities, n_topics, n_subtopics, n_workers):
    predictors = ActorPool([PredictionAgent.remote() for _ in range(n_workers)])

    print('Generating activities')
    predictors.submit(lambda a, v: a.predict_activities.remote(n_activities), None)
    activities = predictors.get_next()

    print('Generating activity topics')
    pbar = tqdm(total=n_activities)
    all_activity_topics = []
    for topics in predictors.map(
        lambda a, v: a.predict_topics.remote(n_topics, v),
        activities
    ):
        pbar.update(1)
        all_activity_topics.append(topics)

    print('Generating subtopics')
    pbar = tqdm(total=n_activities * n_topics)
    all_activity_subtopics = []
    for activity, activity_topics in zip(activities, all_activity_topics):
        activity_topic_subtopics = []
        for subtopics in predictors.map(
            lambda a, v: a.predict_subtopics.remote(n_subtopics, f'{activity} / {v}'),
            activity_topics
        ):
            pbar.update(1)
            activity_topic_subtopics.append(subtopics)
        all_activity_subtopics.append(activity_topic_subtopics)

    with open(output_file, 'wb') as outfile:
        pickle.dump([activities, all_activity_topics, all_activity_subtopics], outfile)


if __name__ == '__main__':
    main()