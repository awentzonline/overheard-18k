# come up with conversations that occur in real life
from functools import partial
import os
import pickle
import random

import click
from dsp import Claude
import dspy
from pydantic import BaseModel, Field
import ray
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

model = Claude(model="claude-3-5-sonnet-20240620", max_tokens=4096)
dspy.settings.configure(lm=model)


class SuggestConversation(dspy.Signature):
    """
    Given the topic at hand, generate a short totally normal conversation between two people.
    The speakers don't necessarially need to greet each other as the conversation may occur at any point in time during their mutual interactions.
    Use realistic personalities. Real interactions range from cheerful and productive to grumpy and unproductive.
    No speaker annotations.
    Around 200 words.
    Only JSON no commentary.
    """
    topic: str = dspy.InputField(desc="Topic at hand")
    lines: list[str] = dspy.OutputField()


@ray.remote
class PredictionAgent:
    def __init__(self):
        self.model = Claude(model="claude-3-5-sonnet-20240620", max_tokens=4096, temperature=0.3)
        dspy.settings.configure(lm=self.model)
        self.pred_conversation = dspy.TypedPredictor(SuggestConversation)

    def predict_conversation(self, full_topic):
        try:
            return self.pred_conversation(topic=full_topic).lines
        except Exception as e:
            print('pred conv problem')
            print(self.model.inspect_history(n=10))
            raise


@click.command()
@click.option('--conv-file', default='conversations.pkl')
@click.option('--num-convs', default=1)
@click.option('--save-steps', default=1000)
@click.option('--n_workers', type=int, default=10)
def main(conv_file, num_convs, save_steps, n_workers):
    with open('topics.pkl', 'rb') as infile:
        activities, all_activity_topics, all_activity_subtopics = pickle.load(infile)

    predictors = ActorPool([PredictionAgent.remote() for _ in range(n_workers)])

    conversations = []
    seen_ids = set()
    # Try loading up any previously generated conversations
    if conv_file and os.path.exists(conv_file):
        try:
            with open(conv_file, 'rb') as infile:
                conversations = pickle.load(infile)
        except Exception as e:
            print('Could not open conversation file:', conv_file)
            print(e)
        else:
            print(f'Found {len(conversations)} conversations complete')
            for i, j, k, conv in conversations:
                row_id = (i, j, k)
                seen_ids.add(row_id)

    print('Generating conversations')
    pbar = tqdm(
        total=len(all_activity_subtopics) * len(all_activity_subtopics[0]) * len(all_activity_subtopics[0][0])- len(conversations)
    )
    row_ids = []
    row_i = len(seen_ids)
    for act_i, activity in enumerate(activities):
        for topic_i, topic in enumerate(all_activity_topics[act_i]):
            for subtopic_i, subtopic in enumerate(all_activity_subtopics[act_i][topic_i]):
                # print(activity, topic, subtopic)
                row_id = (act_i, topic_i, subtopic_i)
                row_ids.append(row_id)
                if row_id in seen_ids:
                    continue  # dont need to update seen_ids because it doesn't help after initial scan
                full_topic = ' / '.join((activity, topic, subtopic))
                predictors.submit(lambda a, v: a.predict_conversation.remote(v), full_topic)

    print('Waiting on jobs...')
    try:
        while predictors.has_next():
            conversation = predictors.get_next()
            row_id = row_ids[row_i]
            row_i += 1
            conversations.append(
                [*row_id, conversation]
            )
            print(conversation)
            pbar.update(1)
            if save_steps and random.uniform(0, 1) < 1. / save_steps:
                print('Saving progress...')
                with open(conv_file, 'wb') as outfile:
                    pickle.dump(conversations, outfile)
    except KeyboardInterrupt:
        print('Stopping..')

    with open(conv_file, 'wb') as outfile:
        pickle.dump(conversations, outfile)


if __name__ == '__main__':
    main()