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

*Play around with the attention matrices inside the model when I get a chance.*

We leave this here, and move onto a beefier training run.


## 2. A slightly more coherent GPT: attempting the full OpenWebText dataset, on mulitple GPUs.


