# pytorch-BYOL

A pytorch implementation of BYOL(Bootstrap Your Own Latent) [Paper link](https://papers.nips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)

## Benchmarks on STL10

| Backbone | Batch size | n_devices |   lr  | epochs | n_hidden | n_output_channel | Image shape |  tau  | Linear eval acc. |
|:--------:|:----------:|:---------:|:-----:|:------:|:--------:|:----------------:|:-----------:|:-----:|:----------------:|
| Resnet18 |     64     |     2     | 0.015 |   40   |    512   |        128       |   (96, 96)  | 0.996 |      0.7029      |
| Resnet18 |     64     |     2     | 0.015 |   80   |    512   |        128       |   (96, 96)  | 0.996 |      0.7655      |
|          |            |           |       |        |          |                  |             |       |                  |