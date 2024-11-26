# GPT from scratch

This is a collection of scripts exploring various aspects of autoregressive language modelling. Planned topics are:

- Basic reimplementation of GPT using a character level tokenizer, and training it on the simple 'tiny shakespeare' dataset
- Implementing a version of the GPT tokenizer and exploring tokenization schemes
- Pretraining GPT: we follow Karpathy's repo
- Some barebones reimplementation of RLHF

## 0. Reimplementing GPT on Shakespeare

We (pre)train for 5 epochs, and saw the following improvement:

```
Epoch 0 train loss: 1.5968784672840788
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [04:26<00:00, 58.94it/s]
what're we outputting?

O:
Cans, Well, both sweeter than I may be:
This good exempt.

Second Citizen:
Why douest God show?                                                          
Epoch 1 train loss: 1.4028351123978235
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [04:21<00:00, 59.90it/s]
what're we outputting?                                           
His do with though all. We can be but under a hour;                                      
Like uddering to them; I'll be forth an alleger-
Epoch 2 train loss: 1.3635213571176963
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [04:21<00:00, 60.03it/s]
what're we outputting?                                            
Come most queen! see. Lead us by this,                                                        
That my brother stol with us this body,                                                        
As you, and thus firs                                                                                                                
Epoch 3 train loss: 1.3413858302961232                                                         
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15685/15685 [04:21<00:00, 60.09it/s]
what're we outputting?                                                                                   
KING RICHARD II:                                        
And stand 'em, all excout him, as not the face                                                                                     
Thesen be a grantil, and the gentlew                                                                                       
Epoch 4 train loss: 1.326011501059321 
```
We see rapid improvement over just a few epochs, however we see many words are misspelt and so on -- because we are using a character-level tokenizer. We next move on to implementing a byte-pair encoding tokenizer and perform he same analysis, but on a larger dataset.

## 1. BPE tokenizer for GPT 2

We re-implemented the bpe tokenizer for GPT2 within the `bpe_tokenizer.py` script.

## 2. A small, incoherent GPT: first attempts

As is well-known, a GPT model requires lots and lots of both compute as well as data. However, we decided to try and finetune a very simple, small and scaled down GPT model to see how it performs. We trained our model on a *scaled down* version of the original [OpenWebText](https://huggingface.co/datasets/stas/openwebtext-10k) dataset for one epoch.

 These are the hyperparamters and hardware settings we used:

- NVIDIA V100
- The following training hyperparameters:
```
grad_accum_steps = 16
train_batch_size = 32
eval_batch_size = 32
block_size = 512
```
- And this is the GPT config used:
```
model = GPT(num_blocks=2,
              vocab_size=50257,
              seq_len=block_size,
              num_heads=4,
              head_dim=4,
              dropout=0.0,
              embedding_dim=768
            )
```
- We used a (somewhat arbitrarily chosen) starting learning rate of `5e-4`, cosine-annealed with a mininum learning rate of `6e-4`.

We noticed that our tiny model could hardly generalise well due to the tiny dataset it was trained on. So we tried to generate some outputs by prompting our LM from a sentence from the training set. The prompt is `Many of those live in the developing countries with limited network connectivity or limited data plan.`. Here's what we got:

- `'Many of those live in the developing countries with limited network connectivity or limited data plan.\n\nThe report is a tool for the technical and the ability to track record of TriMet females, which are more likely to rate large categories of the advances in the form of data.\n\nOver the past five years, the study tracked changes in the world. One study concluded that the paper’s authors successfully increased their sample to 85 percent, compared with 85 percent, and 35 percent of the authors, compared with 32 percent male.\n\npeople with sexually selective data from pregnancy ('`
- ` "Many of those live in the developing countries with limited network connectivity or limited data plan.\n\nThe new version of the data from the Internet of Things. It's not a fairly delicate balance sheet for a lot of data usage. This also means that the data is a compromise when the data center of the data center of the PRISM program. The lists are meant to be reliable.\n\nThe new version of the same site features that are published back-end database and the query tracks and the input element. The following approach was created by the writes checks the metadata to the input"`
- `"Many of those live in the developing countries with limited network connectivity or limited data plan.\n\nThe new report is to the data generated by the Sky News Service (SWIFT) to work with the 'Timth' is not supported by the 'mass board' attacks' attacks that the 'horizontal' ' quote that the discjan scenario used for the original cluster of the original camera fixed and the original camera as the original AR/3d3d3d3:1. The broadcast is the equivalent of the square root of the unit, and the unit that the unit"`

Still quite incoherent! But, consistently, we notice that the model is able to recognise the word `data`, and, even if it doesn't use it properly when it continues, it's able to dig into its memory to find some meaning in the word. This suggests that the attention mechanism is working and that it's able to pay attention to semantically relevant words.

We find that using a temperature of 0.3 produces the least chaotic outputs.

*Play around with the attention matrices inside the model when I get a chance.*

We leave this here, and move onto a beefier training run.


## 2. A slightly more coherent GPT: attempting the full FineWeb dataset.

## Single GPU
We ended up training our GPT model on the [edu-fineweb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset, as in Karpathy's lecture. Training for one epoch on a single A10G GPU took about 60 hours.

 According to [Karpathy](https://x.com/karpathy/status/1795513568655487221?prefetchTimestamp=1732596994254), we were able to (marginally) surpass the performance of the original GPT2 model because edu-fineweb is a better-curated dataset than the original (unreleased) dataset that GPT2 was trained on. To evaluate, we simply used the [hellawag](https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py) script from the `build-nanogpt` repo. 

 We got the following hellaswag eval results:

Ours:
 ```
10042 acc_norm: 3060/10042=0.3047
 ```

 GPT2 Baseline:
 ```
10042 acc_norm: 2968/10042=0.2956
 ```

 Sample generation looks like (prompted with `"Hello, I am a langauge model,"`)

```
 sample 0: Hello, I am a language model, it was not done for people. There were a lot of students to learn this language. I learnt most of them. One of them said… "I know how to speak English and how to use English in my life, if I could speak a second language. It will take me more time to do more. We never ask for it, we don't even hear about it, I get it. I'll learn it and my

sample 1: Hello, I am a language model, and I have no knowledge of the world. No. I am just having an interest in the technology. It is amazing. And I wonder why I have been there, and not the people who see me, and who know I have just gone out and looked at me - and they know me because we know each other intimately and I share their feelings a lot. I will be there for ever, and I must say thank you.
A few weeks ago
```

Certainly more coherent, but far from the state-of-the-art (even GPT3, which doesn't use RLHF is far better). As always, more data and more epochs will aid with performance. The purpose of this repo is to get a feel for what pretraining an LLM is actually like, and what the challenges actually are. As we've seen, it's clear that to even get somewhat coherent results LLMs need an enormous amount of data. As a simple individual limited on resources and money, getting a functioning model is quite tough!

To run the single GPU version, first generate the tokenized data shards with `python tokenize_dataset.py`. 100 shards, each with 100M tokens (save for the last (validation) shard which will have less) and will be saved to your local file system under `./edu_fineweb`. Then simply run `python train_gpt_single_gpu.py`, but in line 56 make sure to put the correct path to your dataset shards. After an eternity (and your hair turning first grey then white), you'll see a model checkpoint pop out and you can evaluate it just like above with `python hellawsag.py`. Just be sure to uncomment the lines betwen `### MY TRAINED MODEL ###` and you're good to go. To generate outputs, use `python generate_sequences.py`.


## Multi GPU

