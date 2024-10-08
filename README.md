# GPT from scratch

This is a collection of scripts exploring various aspects of autoregressive language modelling. Planned topics are:

- Basic reimplementation of GPT using a character level tokenizer, and training it on the simple 'tiny shakespeare' dataset
- Implementing a version of the GPT tokenizer and exploring tokenization schemes
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

We're a ragtag set of individuals with limited GPU access so please don't penalise us for not being able to be fancy! ~ goodday

