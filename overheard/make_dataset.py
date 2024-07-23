import json
import pickle

import click
from tqdm import tqdm


@click.command()
@click.argument('topic_file')
@click.argument('conversation_file')
@click.argument('output_file')
def main(topic_file, conversation_file, output_file):
    with open(topic_file, 'rb') as infile:
        activities, all_activity_topics, all_activity_subtopics = pickle.load(infile)

    with open(conversation_file, 'rb') as infile:
        conversations = pickle.load(infile)

    outfile = open(output_file, 'w')
    print(f'Found {len(conversations)} conversations')
    for i, j, k, conversation in tqdm(conversations):
        conversation = '\n'.join(conversation)
        outfile.write(json.dumps(dict(
            text=conversation,
            activity=activities[i],
            topic=all_activity_topics[i][j],
            subtopic=all_activity_subtopics[i][j][k]
        )))
    outfile.close()

if __name__ == '__main__':
    main()