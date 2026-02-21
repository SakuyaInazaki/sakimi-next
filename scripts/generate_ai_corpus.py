#!/usr/bin/env python3
"""
High-diversity Chinese synthetic corpus generator.

Design goals:
- Avoid rigid templates from earlier versions.
- Enforce quality gates (length, Chinese ratio, repetition, banned phrases).
- Enforce diversity gates (prefix cap, SimHash near-dedup).
- Keep outputs compatible with existing Rust data pipeline (one sample per line).
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BANNED_PHRASES = [
    "作为AI",
    "作为一个AI",
    "我是AI",
    "我不能",
    "首先，",
    "其次，",
    "最后，",
    "总而言之",
    "综上所述",
]

STYLE_POOL = [
    "科普说明",
    "案例拆解",
    "工作手记",
    "问答解释",
    "流程指南",
    "对比分析",
    "复盘总结",
    "政策解读",
    "现场观察",
    "实践经验",
]

OPENERS = [
    "围绕{topic}，真正决定效果的往往不是单点技术，而是系统协作。",
    "讨论{topic}时，最常见的误区是只盯结果，不看过程。",
    "把{topic}做好，通常要先把问题拆小，再把责任落到具体环节。",
    "在现实场景里，{topic}并不神秘，它更像一门持续迭代的工程。",
    "很多团队在推进{topic}时起步很快，但稳定运行阶段才见功力。",
    "如果把{topic}当作一次性任务，后续很容易在细节处失速。",
    "对{topic}的理解越深入，越会意识到标准化和弹性要同时存在。",
    "评价{topic}是否成熟，关键不在口号，而在可重复执行的机制。",
    "面对{topic}，最值得先做的是统一目标和约束条件。",
    "在多数行业实践里，{topic}的难点从来不是开始，而是持续做好。",
]

TRANSITIONS = [
    "进一步看",
    "换个角度",
    "在实践中",
    "从执行层面看",
    "同时",
    "另外",
    "更重要的是",
    "在复盘阶段",
    "把时间拉长后",
    "从组织协同看",
    "就一线经验而言",
    "在约束条件变化时",
    "在跨团队场景下",
]

RISKS = [
    "指标定义不一致",
    "反馈延迟过高",
    "任务边界模糊",
    "角色责任交叉",
    "异常上报链路过长",
    "短期冲刺挤压长期建设",
    "数据采集口径频繁变化",
    "只做汇报不做闭环",
    "上线节奏与保障能力失配",
    "把个别经验误当通用规律",
]

MITIGATIONS = [
    "建立固定的异常分级规则",
    "把关键字段沉淀为统一模板",
    "为核心流程设置可观测里程碑",
    "在每个周期做一次小步回看",
    "设置双人复核避免口径漂移",
    "用样本复盘替代空泛讨论",
    "把高频故障写成标准处置清单",
    "将决策依据与结果绑定存档",
    "按影响范围设计应急预案",
    "先明确停机线再做优化尝试",
]

ACTIONS = [
    "梳理",
    "校准",
    "验证",
    "对齐",
    "拆解",
    "归并",
    "补齐",
    "稳住",
    "压缩",
    "细化",
    "联动",
    "追踪",
    "盘点",
    "优化",
]

OBJECTS = [
    "流程",
    "指标",
    "策略",
    "资源",
    "节奏",
    "接口",
    "样本",
    "规则",
    "记录",
    "机制",
    "路径",
    "边界",
]

SCENES = [
    "月度复盘",
    "上线前夜",
    "跨部门周会",
    "现场巡检",
    "高峰时段",
    "灰度发布窗口",
    "版本切换期",
    "业务回暖阶段",
    "突发告警后",
    "新成员交接时",
]

NARRATIVE_ROLES = [
    "夜班工程师",
    "社区医生",
    "仓库调度员",
    "地铁司机",
    "图书管理员",
    "乡村教师",
    "售后班组长",
    "值班运维",
    "项目协调员",
    "实习生",
]

NARRATIVE_EVENTS = [
    "临时任务叠加",
    "核心设备告警",
    "排班突发变动",
    "客户集中反馈",
    "关键物资短缺",
    "接口响应异常",
    "上游交付延期",
    "天气导致外部延误",
]

QA_QUESTION_PREFIX = [
    "问：",
    "问题：",
    "常见问题：",
]


@dataclass
class DomainConfig:
    weight: float
    topics: List[str]


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "science_encyclopedia": DomainConfig(
        weight=0.35,
        topics=[
            "城市交通调度",
            "土壤改良",
            "微生物发酵",
            "电池寿命管理",
            "海绵城市建设",
            "低温保鲜流程",
            "农业灌溉优化",
            "公共图书馆服务",
            "河道生态修复",
            "工业质量追踪",
            "基础统计建模",
            "智能家居联动",
            "中小企业库存管理",
            "空气质量监测",
            "校园课程设计",
            "社区养老服务",
            "医疗分诊流程",
            "供应链协同",
            "应急演练组织",
            "能源负荷预测",
            "慢病随访管理",
            "旧城更新项目",
            "车间节拍优化",
            "数字档案建设",
            "污水处理调优",
            "冷链配送保障",
            "应急通信保障",
            "教学质量评估",
            "基层健康宣教",
            "中台数据治理",
        ],
    ),
    "practical_writing": DomainConfig(
        weight=0.25,
        topics=[
            "会议纪要写作",
            "需求文档拆解",
            "周报模板优化",
            "新人培训手册",
            "跨部门沟通",
            "线上活动执行",
            "客服话术整理",
            "采购流程规范",
            "数据报表复盘",
            "项目风险登记",
            "故障排查记录",
            "值班交接说明",
            "产品上线清单",
            "质量验收标准",
            "代码评审流程",
            "运维变更通知",
            "季度目标对齐",
            "供应商沟通模板",
            "服务SLA说明",
            "巡检报告模板",
        ],
    ),
    "narrative_story": DomainConfig(
        weight=0.20,
        topics=[
            "夜班协作",
            "社区服务",
            "校务支持",
            "仓储轮转",
            "应急响应",
            "一线排障",
            "节前保供",
            "新人成长",
            "站点联动",
            "跨班交接",
            "项目救火",
            "季节性保障",
            "窗口期部署",
            "风险回落",
        ],
    ),
    "qa_explainer": DomainConfig(
        weight=0.20,
        topics=[
            "为什么要做版本管理",
            "如何规划学习路径",
            "为什么要写复盘",
            "怎样提高专注力",
            "如何理解概率思维",
            "怎样建设团队协作",
            "为什么要做自动化测试",
            "如何降低线上故障率",
            "怎样评估需求优先级",
            "为什么要关注数据质量",
            "如何安排长期训练计划",
            "怎样做有效沟通",
            "为什么要做容量规划",
            "如何避免低效会议",
            "怎样建立知识库",
            "如何设计验收标准",
            "为什么要做灰度发布",
            "如何处理冲突需求",
        ],
    ),
}


def pick_domain(rng: random.Random) -> str:
    domains = list(DOMAIN_CONFIGS.keys())
    weights = [DOMAIN_CONFIGS[d].weight for d in domains]
    return rng.choices(domains, weights=weights, k=1)[0]


def parse_domain_ratios(raw: str) -> Dict[str, float]:
    """
    Parse domain ratios from string:
    "science_encyclopedia:0.35,practical_writing:0.25,narrative_story:0.2,qa_explainer:0.2"
    """
    ratios: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid ratio segment: {part}")
        name, value = part.split(":", 1)
        name = name.strip()
        if name not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain in ratios: {name}")
        ratios[name] = float(value.strip())

    if not ratios:
        raise ValueError("Empty domain ratios")

    # Fill missing domains using configured default weights.
    for domain, cfg in DOMAIN_CONFIGS.items():
        if domain not in ratios:
            ratios[domain] = cfg.weight

    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Domain ratios must sum to positive value")

    # Normalize to sum=1.
    for key in list(ratios.keys()):
        ratios[key] = ratios[key] / total

    return ratios


def compute_domain_targets(samples: int, ratios: Dict[str, float]) -> Dict[str, int]:
    domain_order = list(DOMAIN_CONFIGS.keys())
    raw_targets = {d: samples * ratios[d] for d in domain_order}
    targets = {d: int(raw_targets[d]) for d in domain_order}
    assigned = sum(targets.values())

    # Distribute remainder by largest fractional parts.
    remainder = samples - assigned
    frac_order = sorted(
        domain_order,
        key=lambda d: (raw_targets[d] - int(raw_targets[d])),
        reverse=True,
    )
    for i in range(remainder):
        targets[frac_order[i % len(frac_order)]] += 1

    return targets


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return zh / len(text)


def repeated_ngram_ratio(text: str, n: int = 10) -> float:
    if len(text) < n * 2:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    freq = Counter(grams)
    repeated = sum(v - 1 for v in freq.values() if v > 1)
    return repeated / len(grams)


def unique_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(set(text)) / len(text)


def sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[。！？!?]", text) if p.strip()]
    return len(parts)


def normalize_text(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"([。！？])\1+", r"\1", text)
    text = text.replace("在在", "在")
    text = text.replace("的的", "的")
    text = text.replace("，，", "，")
    text = text.replace("。。", "。")
    if text and text[-1] not in "。！？":
        text += "。"
    return text


def text_hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def simhash64(text: str, ngram: int = 3) -> int:
    text = re.sub(r"\s+", "", text)
    if len(text) < ngram:
        return text_hash64(text)
    feats = Counter(text[i : i + ngram] for i in range(len(text) - ngram + 1))
    bits = [0] * 64
    for feat, w in feats.items():
        h = text_hash64(feat)
        for i in range(64):
            if (h >> i) & 1:
                bits[i] += w
            else:
                bits[i] -= w
    out = 0
    for i, v in enumerate(bits):
        if v >= 0:
            out |= 1 << i
    return out


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimhashIndex:
    def __init__(self, prefix_bits: int = 16, hamming_threshold: int = 2):
        self.prefix_bits = prefix_bits
        self.hamming_threshold = hamming_threshold
        self.buckets: Dict[int, List[int]] = {}

    def _bucket_key(self, h: int) -> int:
        return h >> (64 - self.prefix_bits)

    def is_near_duplicate(self, h: int) -> bool:
        bucket = self._bucket_key(h)
        for existing in self.buckets.get(bucket, []):
            if hamming_distance(existing, h) <= self.hamming_threshold:
                return True
        return False

    def add(self, h: int) -> None:
        bucket = self._bucket_key(h)
        self.buckets.setdefault(bucket, []).append(h)


def random_fact(topic: str, rng: random.Random) -> str:
    x, y = rng.sample(OBJECTS, 2)
    action = rng.choice(ACTIONS)
    scene = rng.choice(SCENES)
    return (
        f"在{scene}，团队通常先{action}{x}，再根据反馈调整{y}，"
        f"这样可以把{topic}中的隐性问题提前暴露。"
    )


def make_science(topic: str, rng: random.Random) -> str:
    style = rng.choice(STYLE_POOL)
    t1, t2, t3 = rng.sample(TRANSITIONS, 3)
    r1, r2 = rng.sample(RISKS, 2)
    m1, m2 = rng.sample(MITIGATIONS, 2)
    a1, a2, a3 = rng.sample(ACTIONS, 3)
    o1, o2, o3 = rng.sample(OBJECTS, 3)

    lines = [
        rng.choice(OPENERS).format(topic=topic),
        rng.choice(
            [
                f"采用{style}视角时，可把{topic}拆成目标定义、执行动作与反馈闭环三个层次。",
                f"若按{style}方式分析，{topic}通常可分为目标设定、动作编排和结果回看三段。",
                f"从{style}出发，处理{topic}时更适合先看目标，再看动作，最后看闭环证据。",
                f"以{style}为框架，{topic}可被拆解为目标、执行与复盘三个连续模块。",
            ]
        ),
        rng.choice(
            [
                f"{t1}，先明确{a1}{o1}的边界，再设置两到三组可比较指标，避免只看单一结果。",
                f"{t1}，应先把{o1}边界讲清，再建立可比指标，避免讨论停留在感受层面。",
                f"{t1}，先完成{o1}口径对齐，再配套对照指标，才能稳定判断优化是否有效。",
                f"{t1}，把{o1}边界与验收条件同时写明，比事后补解释更可靠。",
            ]
        ),
        rng.choice(
            [
                f"{t2}，执行阶段要同步记录异常上下文，特别是触发条件、处置动作与回落时间。",
                f"{t2}，推进过程中要留下异常证据，包括触发原因、处理路径和恢复时点。",
                f"{t2}，建议边执行边记录关键事件，不要把异常细节留到事后补写。",
                f"{t2}，若缺少异常上下文记录，后续复盘往往难以还原真实因果。",
            ]
        ),
        random_fact(topic, rng),
        rng.choice(
            [
                f"常见风险之一是{r1}，另一个高频问题是{r2}，两者叠加时会放大波动。",
                f"一线场景里，{r1}和{r2}经常同时出现，并且会相互放大影响。",
                f"如果{r1}长期得不到治理，通常会与{r2}形成连锁反应。",
                f"经验显示，{r1}并不孤立，它往往与{r2}共同拉高不确定性。",
            ]
        ),
        rng.choice(
            [
                f"为降低波动，建议{m1}，并在关键节点{a2}{o2}，保证口径一致。",
                f"要压低波动，可先{m1}，再在关键节点持续{a2}{o2}。",
                f"想把偏差控制住，通常需要先{m1}，并同步{a2}{o2}的执行证据。",
                f"稳定性提升的前提是{m1}，同时配套{a2}{o2}来固化动作。",
            ]
        ),
        rng.choice(
            [
                f"当环境变化较快时，再配合{m2}，通常能把不确定性控制在可接受范围。",
                f"若外部条件频繁变化，叠加{m2}往往能显著降低决策抖动。",
                f"在高波动阶段，配合{m2}可以让系统回到可管理区间。",
                f"环境扰动加剧时，{m2}通常是维持节奏的重要抓手。",
            ]
        ),
        rng.choice(
            [
                f"最终目标不是追求一次性最优，而是让{topic}在长期运行中保持可解释、可迁移与可迭代。",
                f"评价{topic}是否成熟，不看一次性成绩，而看长期是否可解释、可复用、可迭代。",
                f"{topic}真正的价值来自稳定运行能力，而非短期冲刺后的局部高点。",
                f"从长期收益看，{topic}应追求的是稳定演进，而不是单次峰值表现。",
            ]
        ),
        rng.choice(
            [
                f"如果每轮复盘都能沉淀{a3}{o3}的证据链，下一周期的决策质量会稳定提高。",
                f"每轮复盘都留下{a3}{o3}的证据，下一阶段就更容易做出高质量判断。",
                f"当{a3}{o3}逐步制度化后，团队的决策误差通常会持续下降。",
                f"持续沉淀{a3}{o3}相关证据，能让后续决策更快更稳。",
            ]
        ),
    ]
    return "".join(lines)


def make_practical(topic: str, rng: random.Random) -> str:
    steps = ["第一步", "第二步", "第三步", "第四步", "第五步"]
    rng.shuffle(steps)
    a = rng.sample(ACTIONS, 5)
    o = rng.sample(OBJECTS, 4)
    t = rng.sample(TRANSITIONS, 3)
    risk = rng.choice(RISKS)
    mitigation = rng.choice(MITIGATIONS)

    lines = [
        rng.choice(
            [
                f"围绕{topic}，建议使用可复用的轻量执行框架，而不是临时拼接动作。",
                f"处理{topic}时，最稳妥的起点是先搭建可复用框架，避免靠临场发挥推进。",
                f"{topic}要想稳定交付，通常需要固定流程骨架，而非每次重写执行路径。",
                f"在{topic}场景中，先建立标准化框架，往往比直接冲执行更有效。",
            ]
        ),
        rng.choice(
            [
                f"{steps[0]}先统一输入输出与验收口径，避免后续讨论偏离目标。",
                f"{steps[0]}先把输入、输出和验收条件写清，后续争议会少很多。",
                f"{steps[0]}先对齐目标与验收口径，能够显著减少返工。",
                f"{steps[0]}先明确边界和交付标准，才能保证讨论围绕同一目标展开。",
            ]
        ),
        rng.choice(
            [
                f"{steps[1]}把关键事项拆成可追踪清单，围绕{o[0]}和{o[1]}设置责任人。",
                f"{steps[1]}将关键任务拆成清单化动作，并为{o[0]}与{o[1]}分别指定负责人。",
                f"{steps[1]}把任务拆到可追踪粒度，并围绕{o[0]}、{o[1]}建立责任映射。",
                f"{steps[1]}先形成任务列表，再按{o[0]}和{o[1]}分派角色与时点。",
            ]
        ),
        rng.choice(
            [
                f"{steps[2]}推进过程中持续{a[0]}现场信息，并用最小代价完成{a[1]}。",
                f"{steps[2]}执行阶段持续{a[0]}一线信号，再以低成本完成{a[1]}。",
                f"{steps[2]}推进时要边做边{a[0]}，同时把{a[1]}控制在可承受成本内。",
                f"{steps[2]}在推进链路中持续补充现场信息，并及时{a[1]}关键动作。",
            ]
        ),
        rng.choice(
            [
                f"{t[0]}，如果跨团队协作较多，应固定同步节奏并保留简明状态看板。",
                f"{t[0]}，跨团队场景下最好固定同步节奏，并保持状态看板简洁透明。",
                f"{t[0]}，协作方较多时应建立固定沟通窗口，避免信息碎片化。",
                f"{t[0]}，若参与方较多，保持节奏稳定和看板清晰比频繁开会更重要。",
            ]
        ),
        rng.choice(
            [
                f"{steps[3]}在周期中段对偏差做一次集中处理，重点看{o[2]}是否出现堆积。",
                f"{steps[3]}在中段安排一次偏差清理，重点检查{o[2]}是否开始积压。",
                f"{steps[3]}周期中段应集中处理偏差，避免{o[2]}问题拖到末期放大。",
                f"{steps[3]}建议在中段做一次系统性纠偏，重点关注{o[2]}的连锁影响。",
            ]
        ),
        rng.choice(
            [
                f"{steps[4]}复盘时明确哪些动作可标准化，哪些动作需要保留弹性。",
                f"{steps[4]}复盘阶段要区分可固化动作和需保留弹性的环节。",
                f"{steps[4]}收尾复盘应同时回答两件事：哪些可标准化，哪些必须留白。",
                f"{steps[4]}在复盘中明确固化与弹性边界，后续执行会更稳定。",
            ]
        ),
        rng.choice(
            [
                f"实务中最怕的是{risk}，因此要提前{mitigation}。",
                f"经验上，{risk}是高频风险之一，所以需要提前{mitigation}。",
                f"如果忽略{risk}，后续很容易被动应对，因此建议尽早{mitigation}。",
                f"面对{risk}，最有效的策略通常是提前{mitigation}并持续跟踪。",
            ]
        ),
        rng.choice(
            [
                f"{t[1]}，把经验沉淀成模板后，新成员也能更快理解上下文。",
                f"{t[1]}，经验形成模板后，成员切换和交接的成本会明显下降。",
                f"{t[1]}，当经验被结构化沉淀，新人进入项目的学习曲线会更平滑。",
                f"{t[1]}，知识模板化之后，团队协同的一致性通常更高。",
            ]
        ),
        rng.choice(
            [
                f"{t[2]}，只要持续{a[2]}、{a[3]}并适度{a[4]}，{topic}的交付质量通常会明显改善。",
                f"{t[2]}，持续{a[2]}与{a[3]}，再配合适度{a[4]}，{topic}的稳定性会持续提升。",
                f"{t[2]}，当{a[2]}、{a[3]}形成常规动作后，{topic}交付质量往往更可控。",
                f"{t[2]}，把{a[2]}和{a[3]}做成日常动作，并谨慎{a[4]}，通常能稳步改善交付结果。",
            ]
        ),
    ]
    return "".join(lines)


def make_narrative(topic: str, rng: random.Random) -> str:
    role = rng.choice(NARRATIVE_ROLES)
    event = rng.choice(NARRATIVE_EVENTS)
    scene = rng.choice(SCENES)
    t1, t2 = rng.sample(TRANSITIONS, 2)
    a1, a2, a3 = rng.sample(ACTIONS, 3)
    o1, o2 = rng.sample(OBJECTS, 2)
    risk = rng.choice(RISKS)
    mitigation = rng.choice(MITIGATIONS)

    lines = [
        rng.choice(
            [
                f"{scene}里，负责{topic}的{role}刚结束上一轮任务，就遇到了{event}。",
                f"在{scene}，负责{topic}的{role}还没来得及休整，就碰上了{event}。",
                f"{scene}刚开始，承担{topic}的{role}便收到{event}的通知。",
                f"{scene}中，{role}在推进{topic}时突然遭遇{event}，现场节奏被打乱。",
            ]
        ),
        rng.choice(
            [
                "起初现场有些混乱，因为每个人都掌握了部分信息，却缺少统一判断。",
                "最开始大家意见分散，信息不少，但缺少统一的判断框架。",
                "最难的不是没有数据，而是信息分散导致大家难以迅速达成一致。",
                "现场最初并不平稳，问题在于信息碎片化而不是行动不够快。",
            ]
        ),
        rng.choice(
            [
                f"他先组织大家在十分钟内{a1}{o1}，把事实、假设和待验证项分开记录。",
                f"他先带队在短时间内{a1}{o1}，再把事实与猜测分栏记录，避免混淆。",
                f"第一步是快速{a1}{o1}，并把可确认信息与待验证信息拆开管理。",
                f"他要求先{a1}{o1}，把证据、判断和待办分别落到同一张清单里。",
            ]
        ),
        rng.choice(
            [
                "随后再按影响范围排序，优先处理会连锁放大的环节。",
                "接着按影响面排序，先处理最可能触发连锁反应的节点。",
                "随后团队把问题按影响等级分层，先稳住会外溢的风险点。",
                "下一步是按影响范围重排优先级，把潜在连锁问题放在最前。",
            ]
        ),
        rng.choice(
            [
                f"{t1}，小组成员分别承担沟通、执行和复核职责，协作效率开始回升。",
                f"{t1}，成员分工为沟通、执行和复核三条线，效率很快恢复。",
                f"{t1}，团队把职责切成沟通、处置、校验三部分，协同明显顺畅。",
                f"{t1}，明确分工后，沟通链路缩短，执行速度也开始提升。",
            ]
        ),
        rng.choice(
            [
                f"处理中途，团队意识到{risk}才是根因，于是立即调整方案。",
                f"推进到一半时，大家发现真正的根因是{risk}，随即切换处理路径。",
                f"复核过程中，团队确认{risk}是核心问题，便立刻改用新方案。",
                f"中段排查后，团队判断{risk}才是关键矛盾，于是当场调整动作。",
            ]
        ),
        rng.choice(
            [
                f"{t2}，他们通过{mitigation}稳定了节奏，并同步{a2}{o2}。",
                f"{t2}，团队用{mitigation}把节奏拉回正轨，同时持续{a2}{o2}。",
                f"{t2}，在{mitigation}的支持下，执行节奏恢复，并完成了{a2}{o2}。",
                f"{t2}，他们先{mitigation}，再{a2}{o2}，整体波动明显下降。",
            ]
        ),
        rng.choice(
            [
                f"到收尾阶段，{role}又安排一次短复盘，把可复用经验整理成清单。",
                f"收尾时，{role}组织了短复盘，把当天经验固化为可复用条目。",
                f"在结束前，{role}安排快速复盘，将关键经验写成后续可用清单。",
                f"临近收尾，{role}补了一次复盘，确保经验能够沉淀并传递。",
            ]
        ),
        rng.choice(
            [
                "这次经历没有戏剧化转折，却证明了稳健协作比临场情绪更重要。",
                "这次处置并不惊险，但再次说明稳定协作比情绪化应对更可靠。",
                "过程并不传奇，却足以说明方法稳定比一时冲动更能解决问题。",
                "没有夸张情节，却清楚地验证了稳健协同的价值。",
            ]
        ),
        rng.choice(
            [
                f"第二天再面对类似状况时，团队已经能更快{a3}关键动作并降低沟通摩擦。",
                f"第二天遇到同类问题时，团队已经能更快{a3}关键动作，沟通成本也更低。",
                f"当类似场景再次出现时，团队能迅速{a3}动作链路，响应更从容。",
                f"同类问题再来时，团队已能稳定{a3}关键环节，并把沟通摩擦压到更低。",
            ]
        ),
    ]
    return "".join(lines)


def make_qa(topic: str, rng: random.Random) -> str:
    prefix = rng.choice(QA_QUESTION_PREFIX)
    t1, t2, t3 = rng.sample(TRANSITIONS, 3)
    a1, a2, a3 = rng.sample(ACTIONS, 3)
    o1, o2, o3 = rng.sample(OBJECTS, 3)
    risk = rng.choice(RISKS)
    mitigation = rng.choice(MITIGATIONS)

    lines = [
        f"{prefix}{topic}？",
        rng.choice(
            [
                "答：核心原因通常不是某个技巧本身，而是是否建立了可持续的执行闭环。",
                "答：关键不在某一个技巧，而在有没有形成可持续、可复盘的执行链路。",
                "答：本质上看，决定效果的是闭环能力，而不是单点动作是否漂亮。",
                "答：真正的分水岭在于能否把方法做成闭环，而非依赖临时灵感。",
            ]
        ),
        rng.choice(
            [
                f"很多人一开始会忽略{o1}定义，导致后续比较与判断失真。",
                f"常见问题是忽略{o1}口径，结果后续比较容易失真。",
                f"如果{o1}定义不清，后续判断往往会出现偏差。",
                f"不少场景里，正是因为{o1}未对齐，复盘时才会争议不断。",
            ]
        ),
        rng.choice(
            [
                f"{t1}，更稳妥的做法是先{a1}{o2}，再用小范围试运行验证假设。",
                f"{t1}，可先{a1}{o2}，再通过小范围试运行确认思路是否成立。",
                f"{t1}，建议先{a1}{o2}，随后做小样本验证，避免直接全量推进。",
                f"{t1}，先把{o2}相关动作{a1}清楚，再以试运行方式校验假设。",
            ]
        ),
        rng.choice(
            [
                f"如果只追求短期速度，{risk}会在后期集中出现。",
                f"只看短期速度时，{risk}往往会在后段集中暴露。",
                f"当目标只剩“快”，{risk}通常会在关键节点突然放大。",
                f"若过度追求短期效率，{risk}常常在后期形成连锁问题。",
            ]
        ),
        rng.choice(
            [
                f"{t2}，建议把决策依据和结果一并记录，减少复盘时的信息缺口。",
                f"{t2}，把决策依据与结果同步记录，能显著降低复盘信息缺失。",
                f"{t2}，最好把依据、动作和结果放在同一记录里，便于后续追溯。",
                f"{t2}，若能同步沉淀依据与结果，复盘效率通常会更高。",
            ]
        ),
        rng.choice(
            [
                f"在推进过程中持续{a2}关键信号，能帮助团队更早发现偏差。",
                f"推进中持续{a2}关键迹象，可以更早识别偏差并纠正。",
                f"只要持续{a2}核心信号，很多偏差都能在放大前被拦截。",
                f"保持对关键信号的持续{a2}，是提前止损的有效方式。",
            ]
        ),
        rng.choice(
            [
                f"必要时结合{mitigation}，把不确定性压缩到可控范围。",
                f"必要时叠加{mitigation}，通常能把不确定性拉回可控区间。",
                f"当波动升高时，可用{mitigation}来稳定节奏并降低风险外溢。",
                f"在不确定性较高阶段，配合{mitigation}往往更稳妥。",
            ]
        ),
        rng.choice(
            [
                f"{t3}，当成员都理解{o3}与目标之间的关系，协作成本会明显下降。",
                f"{t3}，一旦成员理解{o3}和目标的关系，协作阻力通常会下降。",
                f"{t3}，把{o3}与目标映射讲清楚，团队协作会更顺畅。",
                f"{t3}，当{o3}与目标的关系被清晰表达，沟通成本会显著降低。",
            ]
        ),
        rng.choice(
            [
                f"长期看，真正有效的是把{topic}沉淀为可复用的方法，而不是依赖个人经验。",
                f"从长期效果看，应把{topic}做成可复用方法，而非依赖个体记忆。",
                f"最终要追求的，是将{topic}转化为团队可继承的工作方法。",
                f"长期而言，把{topic}制度化比依赖个人经验更可靠。",
            ]
        ),
    ]
    return "".join(lines)


def build_sample(domain: str, topic: str, rng: random.Random) -> str:
    if domain == "science_encyclopedia":
        return make_science(topic, rng)
    if domain == "practical_writing":
        return make_practical(topic, rng)
    if domain == "narrative_story":
        return make_narrative(topic, rng)
    return make_qa(topic, rng)


def quality_score(text: str) -> Tuple[int, Dict[str, float], List[str]]:
    metrics = {
        "len": float(len(text)),
        "zh_ratio": chinese_ratio(text),
        "sentences": float(sentence_count(text)),
        "uniq_char_ratio": unique_char_ratio(text),
        "repeat_10gram": repeated_ngram_ratio(text, 10),
        "repeat_6gram": repeated_ngram_ratio(text, 6),
    }

    reasons = []
    score = 0

    if 320 <= metrics["len"] <= 980:
        score += 4
    else:
        reasons.append("length_out_of_range")

    if metrics["zh_ratio"] >= 0.90:
        score += 4
    else:
        reasons.append("zh_ratio_low")

    if 8 <= metrics["sentences"] <= 14:
        score += 4
    else:
        reasons.append("sentence_count_bad")

    if 0.11 <= metrics["uniq_char_ratio"] <= 0.45:
        score += 3
    else:
        reasons.append("char_diversity_bad")

    if metrics["repeat_10gram"] <= 0.015:
        score += 4
    else:
        reasons.append("repeat_10gram_high")

    if metrics["repeat_6gram"] <= 0.06:
        score += 3
    else:
        reasons.append("repeat_6gram_high")

    if not any(p in text for p in BANNED_PHRASES):
        score += 3
    else:
        reasons.append("banned_phrase")

    if re.search(r"(.)\1\1\1", text):
        reasons.append("char_run_high")
    else:
        score += 2

    return score, metrics, reasons


def generate_corpus(
    output_path: Path,
    meta_path: Path,
    samples: int,
    seed: int,
    min_score: int,
    max_prefix_repeat: int,
    prefix_chars: int,
    balanced: bool,
    domain_ratios: Dict[str, float],
    max_attempt_multiplier: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    exact_hashes = set()
    sim_index = SimhashIndex(prefix_bits=16, hamming_threshold=2)
    prefix_counter: Counter[str] = Counter()

    accepted = 0
    rejected = 0
    total_chars = 0
    reject_reasons: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    total_score = 0
    attempts = 0
    max_attempts = max(samples * max_attempt_multiplier, samples + 1)

    domain_targets: Optional[Dict[str, int]] = None
    domain_remaining: Optional[Dict[str, int]] = None
    if balanced:
        domain_targets = compute_domain_targets(samples, domain_ratios)
        domain_remaining = dict(domain_targets)

    with output_path.open("w", encoding="utf-8") as txt_out, meta_path.open(
        "w", encoding="utf-8"
    ) as meta_out:
        while accepted < samples and attempts < max_attempts:
            attempts += 1

            if balanced and domain_remaining is not None:
                available = [d for d, n in domain_remaining.items() if n > 0]
                if not available:
                    break
                # Sample according to remaining quotas to keep exact balance.
                weights = [domain_remaining[d] for d in available]
                domain = rng.choices(available, weights=weights, k=1)[0]
            else:
                domain = pick_domain(rng)

            topic = rng.choice(DOMAIN_CONFIGS[domain].topics)

            text = build_sample(domain, topic, rng)
            text = normalize_text(text)

            score, metrics, reasons = quality_score(text)
            if score < min_score:
                rejected += 1
                if reasons:
                    reject_reasons[reasons[0]] += 1
                continue

            h = text_hash64(text)
            if h in exact_hashes:
                rejected += 1
                reject_reasons["exact_duplicate"] += 1
                continue

            sh = simhash64(text)
            if sim_index.is_near_duplicate(sh):
                rejected += 1
                reject_reasons["near_duplicate"] += 1
                continue

            prefix = re.sub(r"[，。！？!?、；：\s]", "", text[:prefix_chars])
            if prefix_counter[prefix] >= max_prefix_repeat:
                rejected += 1
                reject_reasons["prefix_reuse"] += 1
                continue

            exact_hashes.add(h)
            sim_index.add(sh)
            prefix_counter[prefix] += 1

            txt_out.write(text + "\n")
            meta_out.write(
                json.dumps(
                    {
                        "id": accepted,
                        "domain": domain,
                        "topic": topic,
                        "char_len": len(text),
                        "score": score,
                        "metrics": metrics,
                        "seed": seed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            accepted += 1
            total_chars += len(text)
            total_score += score
            domain_counter[domain] += 1
            if balanced and domain_remaining is not None:
                domain_remaining[domain] -= 1

            if accepted % 5000 == 0:
                print(f"generated {accepted}/{samples}")

    avg_chars = total_chars / accepted if accepted else 0
    avg_score = total_score / accepted if accepted else 0

    return {
        "accepted": accepted,
        "rejected": rejected,
        "accept_rate": accepted / (accepted + rejected) if accepted + rejected else 0,
        "total_chars": total_chars,
        "avg_chars": avg_chars,
        "avg_score": avg_score,
        "domain_counts": dict(domain_counter),
        "top_reject_reasons": reject_reasons.most_common(10),
        "seed": seed,
        "min_score": min_score,
        "max_prefix_repeat": max_prefix_repeat,
        "prefix_chars": prefix_chars,
        "balanced": balanced,
        "domain_targets": domain_targets or {},
        "attempts": attempts,
        "max_attempts": max_attempts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-diversity Chinese corpus")
    parser.add_argument(
        "--output",
        default="data/synth/train_ai_hq.txt",
        help="output text file path",
    )
    parser.add_argument(
        "--meta",
        default="data/synth/train_ai_hq.meta.jsonl",
        help="output metadata jsonl path",
    )
    parser.add_argument(
        "--report",
        default="data/synth/train_ai_hq.report.json",
        help="output report json path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="number of accepted samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260218,
        help="random seed",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=19,
        help="minimum quality score to accept",
    )
    parser.add_argument(
        "--max-prefix-repeat",
        type=int,
        default=40,
        help="max allowed reuse of normalized prefix",
    )
    parser.add_argument(
        "--prefix-chars",
        type=int,
        default=96,
        help="number of leading chars used for prefix reuse control",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="enforce domain ratio by hard quotas",
    )
    parser.add_argument(
        "--domain-ratios",
        default="science_encyclopedia:0.35,practical_writing:0.25,narrative_story:0.2,qa_explainer:0.2",
        help="comma separated domain ratios, e.g. science_encyclopedia:0.35,...",
    )
    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=120,
        help="max attempts = samples * multiplier before abort",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_ratios = parse_domain_ratios(args.domain_ratios)

    report = generate_corpus(
        output_path=Path(args.output),
        meta_path=Path(args.meta),
        samples=args.samples,
        seed=args.seed,
        min_score=args.min_score,
        max_prefix_repeat=args.max_prefix_repeat,
        prefix_chars=args.prefix_chars,
        balanced=args.balanced,
        domain_ratios=domain_ratios,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"output: {args.output}")
    print(f"meta: {args.meta}")
    print(f"report: {args.report}")
    print(f"accepted: {report['accepted']}")
    print(f"rejected: {report['rejected']}")
    print(f"accept_rate: {report['accept_rate']:.4f}")
    print(f"total_chars: {report['total_chars']}")
    print(f"avg_chars: {report['avg_chars']:.2f}")
    print(f"avg_score: {report['avg_score']:.2f}")


if __name__ == "__main__":
    main()
