---
layout: post
title:  "PyTorch + Tensorboard (quick-start)"
date:   2021-05-24 20:41:00 -0400
categories: ML general 
permalink: /:categories/:title:output_ext
---
## Using tensorboard + pytorch in jupyter notebooks
This is a quick-start guide for using tensorboard and pytorch.

### td;dr
Go to [this](https://colab.research.google.com/drive/1t6CdwOV-7_jIUtsw2F6EEjfCQL9bH4iE?usp=sharing) colab notebook for a quick-start guide... or read below. It's the same content!

### What is TensorBoard?
Tensorboard is TensorFlow's visualization toolkit.

> TensorBoard provides the visualization and tooling needed for machine learning experimentation

What is important to understand:

- it allows us to track and visualize metrics/values of the model as they change over time
- it works independently of tensorflow. In fact, we are using it with pytorch here!
- collection/logging is somewhat independent of visualization - they are tied together by the log files/db. Collection/logging populates files/db; visualization (tensorboard) displays data stored in files/db.

### Collecting data (SummaryWriter)
As mentioned before, collection of data is/can be a completely separate step. In fact, we are going to use `SummaryWriter` provided by `torch` to collect data. `SummaryWriter` can be found in `torch.utils.tensorboard.SummaryWriter`

The basic idea is as follows:
1. instantiate a writer from `SummaryWriter` - this allows us to specify the log location.
2. push *tagged* data to our writer via methods like `add_scalar()`

```python
from torch.utils.tensorboard import SummaryWriter

# Run 1
writer1 = SummaryWriter('runs/run_1') 

for n_iter in range(100):
    # signature: writer.add_scalar(tag, scalar_value, global_step)
    writer1.add_scalar('tutorial/line', 2 * n_iter, n_iter)
    writer1.add_scalar('tutorial/poly', n_iter ** 2, n_iter)

writer1.close()

# Run 2
writer2 = SummaryWriter('runs/run_2')

for n_iter in range(100):
    # signature: writer.add_scalar(tag, scalar_value, global_step)
    writer2.add_scalar('tutorial/line', 1.5 * n_iter - 5, n_iter)
    writer2.add_scalar('tutorial/poly', n_iter ** 2 - 100* n_iter, n_iter)

writer2.close()
```
 
Here are a few things to note:
- when instantiating `SummaryWriter`, if a log dir is not specified, the default location (runs/) is used
- as with any file stream, the new data is not pushed immediately to the file
  - closing the writer via `writer.close()` flushes the buffer
  - calling `writer.flush()` also flushes the buffer
- `writer1` and `writer2` will write to different locations: `runs/run_1` and `runs/run_2` respectively

Now that a new file has been created with data for 'tutorial/line' and 'tutorial/poly'. Let's start a tensorboard and visualize it! We can do this by calling:

```python
# load tensorboard extension
%load_ext tensorboard
# display tensorboard
%tensorboard --logdir runs/
```

You should see something similar to the board below:

![Tensorboard displayed in a notebook]({{ '/assets/images/tensorboard_display.png' | relative_url }})

*Fig 1. Tensorboard displayed in a notebook*

