# WoBERT
以词为基本单位的中文BERT（Word-based BERT）

## 详情

https://kexue.fm/archives/7758

## 训练

目前开源的WoBERT是Base版本，在哈工大开源的[RoBERTa-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm)基础上进行继续预训练，预训练任务为MLM。初始化阶段，将每个词用BERT自带的Tokenizer切分为字，然后用字embedding的平均作为词embedding的初始化。模型使用单张24G的RTX训练了100万步（大概训练了10天），序列长度为512，学习率为5e-6，batch_size为16，累积梯度16步，相当于batch_size=256训练了6万步左右。训练语料大概是30多G的通用型语料。

此外，我们还提供了WoNEZHA，这是基于华为开源的[NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)进行再预训练的，训练细节跟WoBERT基本一样。NEZHA的模型结构跟BERT相似，不同的是它使用了相对位置编码，而BERT用的是绝对位置编码，因此理论上NEZHA能处理的文本长度是无上限的。这里提供以词为单位的WoNEZHA，就是让大家多一个选择。

**2021年03月03日：**  新增WoBERT Plus模型，以RoBERTa-wwm-ext为基础，中文MLM式预训练，重新构建词表（比已经开源的WoBERT更完善），30+G语料，maxlen=512，batch_size=256、lr=1e-5训练了25万步（4 * TITAN RTX，累积4步梯度，是之前的WoBERT的4倍），每1000步耗时约1580s，共训练了18天，训练acc约64%，训练loss约1.80。

## 依赖
```bash
pip install bert4keras==0.8.8
```

## 下载

- **WoBERT**: [chinese_wobert_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1BrdFSx9_n1q2uWBiQrpalw), 提取码: kim2
- **WoNEZHA**: [chinese_wonezha_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1ABKwUuIiMEEsRXxxlbyKmw), 提取码: qgkq
- **WoBERT<sup>+</sup>**: [chinese_wobert_plus_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1Ltq3ltQsyBCj56zoOOvI9A), 提取码: aedw
- 
## 引用

Bibtex：

```tex
@techreport{zhuiyipretrainedmodels,
  title={WoBERT: Word-based Chinese BERT model - ZhuiyiAI},
  author={Jianlin Su},
  year={2020},
  url="https://github.com/ZhuiyiTechnology/WoBERT",
}
```

## 联系

邮箱：ai@wezhuiyi.com
追一科技：https://zhuiyi.ai
