# 极验 v4 点选验证码数据集

极验 v4 三类题型（phrase / word / icon）的真实抓取图、人工框选标注和裁剪好的单字样本，可以直接训练字框检测、单字识别和语序模型。

数据全部来自真实站点抓取，不含合成样本。标注是我逐张做的，不是自动生成。

## 里面有什么

| 内容 | 数量 | 说明 |
| --- | --- | --- |
| `raw/phrase/` | 1885 张 | 底图，一张一题 |
| `raw/word/` | 2013 组 | 背景图 + 3 张题面小图 |
| `raw/icon/` | 1161 组 | 背景图 + 3 张题面小图 |
| `images/live4_4char/` | 400 张 | 2026-10 改版后的 4 字题 |
| 人工标注 | **802 张 / 2634 框** | phrase 305、word 100、icon 95、4 字题 302 |
| 裁剪单字 | 1911 张 | 777 个不同字符，含来源标记 |

标注分布：

| 题型 | 原始图 | 人工标注 | 框数 |
| --- | --- | --- | --- |
| phrase（3 字） | 1885 | 305 | 915 |
| word（选字） | 2013 组 | 100 | 300 |
| icon（选图标） | 1161 组 | 95 | 285 |
| phrase（4 字，新字形） | 400 | 302 | 1164 |

**标注花了 10 小时以上**，平均每张 1.2 分钟。极验的字形做了对抗性处理，见下面数据特点一节。

## 目录结构

```
geetest-captcha-dataset/
├── raw/
│   ├── phrase/              1885 张底图
│   ├── word/                captcha_00001_bg.jpg + _ques_0~2.png
│   └── icon/                同上
├── images/
│   └── live4_4char/         4 字题 400 张
├── data/
│   ├── phrase_annotations.jsonl   3 字题：框 + 字
│   ├── boxes_word.jsonl           word：框 + 字
│   ├── boxes_icon.jsonl           icon：框（无字）
│   ├── live4_boxes.jsonl          4 字题：框 + 字 + reviewed 标记
│   ├── live4_phrases.txt          4 字题的语序答案
│   ├── all.txt / local.txt / public.txt   单字索引，按来源分开
│   ├── train.txt / val.txt        单字训练清单（路径 + Tab + 字）
│   └── *_manifest.txt             各题型文件清单
├── text_corpus/             三字词语序语料（train/val + 统计）
├── tools/
│   ├── label_server.py      网页标注工具
│   └── jsonl_to_yolo.py     标注转 YOLO 训练目录
├── LICENSE                  MIT（代码）
└── LICENSE-DATA-CC-BY-4.0.txt   CC BY 4.0（数据）
```

## 标注格式

每行一条记录，一个框对应一个字或一个图标：

```json
{"img": "captcha_00001_bg.jpg",
 "boxes": [{"x": 210, "y": 46, "w": 52, "h": 52, "char": "擦"},
           {"x": 52, "y": 90, "w": 50, "h": 52, "char": "净"}],
 "order": [0, 1],
 "reviewed": true}
```

- `char`：icon 型没有字，留空
- `order`：点击顺序
- `reviewed`：人工核对过才为 true，预填但没核对的记录没有这个字段

`data/phrase_manifest.txt`、`word_manifest.txt`、`icon_manifest.txt` 分别是各题型的文件清单，用来核对抓了多少、标了多少。

## 数据来源

单字样本分两部分，`local.txt` 和 `public.txt` 分开索引，互不混淆：

| 来源 | 样本数 | 唯一字符 | 说明 |
| --- | --- | --- | --- |
| local | 879 | 415 | 本项目抓取并人工框选 |
| public | 1032 | 451 | 第三方公开数据集，见下 |
| 合计 | 1911 | 777（去重） | |

public 部分：

```
数据源：百度 AI Studio 数据集 222386
地址：  https://aistudio.baidu.com/datasetdetail/222386
原始格式：LabelMe JSON（target 框 + char 框）
原始图片：355 张，344×384
```

那个数据集是图选型点选验证码（底部有提示图），和极验的语义语序型不同，但字形渲染风格一致，所以它的 target 框裁出来的单字能直接进训练。原始整图没有保留，只留了裁剪后的单字图。

## 数据特点

### 字形渲染

- 白色粗描边 + 饱和填充，2026-10 之后是这种风格（旧版是彩色细描边）
- 字符随机旋转、缩放、位置错落，通常有 2~3 个字互相压叠
- 背景是风景照，颜色复杂，不能靠颜色阈值分割

### 和标准字形的差异

对抗性渲染让部分部件被改写，标注时要格外注意：

| 干扰 | 表现 |
| --- | --- |
| 三点水 | 冫渲染成 1 点 + 竖线，氵渲染成 2 点 + 竖线 |
| 提手旁 | 底部的提被画成竖线，形似亻 |
| 描边覆盖 | 深蓝 / 红 / 黄多色描边压在笔画边缘上 |

### 已知问题

- `boxes_icon.jsonl` 没有字，只能用来训检测器，不能进识别训练
- 少数记录的字有歧义（比如“好排队伍”和“排好队伍”都讲得通），语序标签按当时看到的面板文案定的
- 4 字题的 `live4_boxes.jsonl` 有 400 条记录，其中 302 条核对过，其余是预填

## 快速开始

```bash
git clone https://github.com/LeapYa/geetest-captcha-dataset
cd geetest-captcha-dataset

# 1) 标注转成 YOLO 训练目录
python tools/jsonl_to_yolo.py --jsonl data/live4_boxes.jsonl \
    --images images/live4_4char --out dataset
# 图片 291 张（train 262 / val 29），框 1164 个

# 2) 起标注工具（要补标或者改标时用）
python tools/label_server.py --host 127.0.0.1 --port 8899 --dataset live4
# 浏览器打开 http://127.0.0.1:8899/
```

标注工具默认账号 `LeapYa`，密码 `Lbl#2026kQ7vx`，正式部署时用 `--users "yourname:yourpass"` 改掉。要在 `DATASETS` 里加自己的数据集，或者把新图片挂进来，见 `tools/label_server.py` 顶部的说明。

## 用这份数据能到什么效果

配合 [geetest-captcha-solver](https://github.com/LeapYa/geetest-captcha-solver) 训练：

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 单字识别 | 95.0% | 留出 240 字 |
| 整句 4 字全对 | 85.0% | 留出 60 张 |
| 语序 | 80.8% | 245 条真实题面 |

评测只用我们自己标的那部分，public 数据只当训练样本，不参与算分。

## 标注工具

`tools/label_server.py` 是网页版框选工具，纯标准库，不需要装依赖：

```bash
python tools/label_server.py --host 127.0.0.1 --port 8899 --dataset live4
```

- 服务端在 `DATASETS` 里声明数据集：图片目录、文件名匹配规则、输出 jsonl
- 只推送没标过的图，已标的不再出现
- 手机上也能用，标完立刻写回服务器
- 保存时写 `reviewed: true`，进度按记录算

## 训练格式

跑过 `tools/jsonl_to_yolo.py` 之后得到标准 YOLO 目录：

```
dataset/
├── images/train/   images/val/
├── labels/train/   labels/val/     每行：0 cx cy w h（0~1 归一化）
└── data.yaml                       path / train / val / names
```

单字识别用 `data/train.txt`、`data/val.txt`，每行是 `图片路径 <TAB> 字`。裁字时往外扩 2 像素，别把描边切掉。

## 许可

| 内容 | 许可 |
| --- | --- |
| 数据（`raw/`、`images/`、`data/`、`text_corpus/`） | [CC BY 4.0](LICENSE-DATA-CC-BY-4.0.txt) |
| 代码（`tools/`） | [MIT](LICENSE) |

使用时保留出处：仓库地址 https://github.com/LeapYa/geetest-captcha-dataset 和作者 LeapYa。

`public` 那部分单字的原始署名随数据一起保留，再分发时请一并带上。

数据集为研究用途整理，其中验证码图片版权归原始站点所有。
