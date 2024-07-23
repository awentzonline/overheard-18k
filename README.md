Overheard-18k: 18,000 Totally Normal Conversations
==================================================
18,000 generated normal conversations you might hear in normal places.

If babies learn language by listening to people speak then maybe we need a dataset like that.
I'm making this as a potential dataset for use in the [BabyLM 2024 Challenge](https://babylm.github.io/)
where the object is to train language models with small datasets of 10 million or 100 million words.

Each sample is a list of dialog lines alternating between speakers.

The dataset was generated as follows:
 * generate a list of common activities where people might talk to each other
 * for each activity, generate a list of related potential conversations
 * go one deeper, for each topic suggest more specific aspects of the conversation
 * generate a "normal" conversation for each (activity, topic, subtopic) of approximately 200 words

The version I'm publishing was generated with `claude-3-5-sonnet-20240620`.
Thanks to [Anthropic](https://www.anthropic.com/) who had generously gifted me API credits as prize for solving a CTF.
