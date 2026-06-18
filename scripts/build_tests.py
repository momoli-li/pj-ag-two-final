"""Build conversation_tests.jsonl: 32 multi-turn scripts."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "conversation_tests.jsonl"

def U(text): return {"role": "user", "text": text}
def A(text): return {"role": "agent", "text": text}

def script(sid, scenario, turns, probes):
    return {"id": sid, "scenario": scenario, "turns": turns, "probes": probes}

def pad(theme):
    pads = {
        "weather":  [U("今天天气怎么样？"), A("可以查询当地天气。")],
        "movie":    [U("最近有什么好电影？"), A("有几部口碑不错。")],
        "food":     [U("中午吃啥好？"), A("简餐沙拉都行。")],
        "stock":    [U("最近股市怎么样？"), A("整体震荡。")],
        "tech":     [U("聊聊大模型最新进展？"), A("有不少新动态。")],
        "music":    [U("推荐首歌吧。"), A("可以试试民谣。")],
        "book":     [U("最近读什么书？"), A("非虚构类不错。")],
        "travel":   [U("国内适合冬天去哪？"), A("哈尔滨、三亚都行。")],
    }
    return pads[theme]

scripts = []

# persona (8)
scripts.append(script("S01", "persona",
    turns=[U("我叫林清雨，是一名上海交大的计算机研三学生。"), A("你好林清雨。"),
        U("我目前在做多模态大模型方向的毕业论文。"), A("好的，记下了。"),
        U("我对宠物毛过敏，之后建议猫咖类活动就不用考虑了。"), A("收到，避开猫咖。"),
        U("我下周打算去成都玩三天。"), A("成都三天。"),
        U("帮我安排一份成都行程，要符合我前面说的所有偏好。")],
    probes=[
        {"type": "consistency", "question": "行程中是否避免了猫咖类活动？",
         "ground_truth": "避免猫咖", "ground_truth_keywords": ["不安排猫咖","避开猫咖","无猫咖","不包含宠物咖啡","不推荐猫咖","避免猫咖","避免宠物咖啡","猫咖"]},
        {"type": "forgetting", "question": "用户的姓名是什么？",
         "ground_truth": "林清雨", "ground_truth_keywords": ["林清雨"]},
    ]))

scripts.append(script("S02", "persona",
    turns=[U("我是张航，某券商的量化研究员。"), A("张航你好。"),
        U("我有严重的乳糖不耐受，任何含奶制品的都不行。"), A("好的，避开乳制品。"),
        U("我最近在筹备结婚，预算比较紧。"), A("理解。"),
        U("帮我推荐三家适合周末家庭聚餐的北京餐厅，要符合我的情况。")],
    probes=[
        {"type": "consistency", "question": "餐厅是否考虑了避开乳制品？",
         "ground_truth": "避开乳制品", "ground_truth_keywords": ["不含乳","无奶酪","避开乳制品","不主打奶","不含奶","避开奶","少奶制品","不推荐奶","乳糖"]},
        {"type": "forgetting", "question": "用户的职业是什么？",
         "ground_truth": "量化研究员", "ground_truth_keywords": ["量化研究员","量化","券商"]},
    ]))

scripts.append(script("S03", "persona",
    turns=[U("我叫王梦琪，今年28岁，住在杭州西湖区。"), A("王梦琪，杭州。"),
        U("我是素食主义者，已经吃素六年了。"), A("素食。"),
        U("我有一只柴犬叫豆豆，3岁。"), A("豆豆，柴犬。"),
        U("帮我策划一次本周末和宠物一起的户外活动，包括午餐建议。")],
    probes=[
        {"type": "consistency", "question": "活动是否考虑素食？",
         "ground_truth": "素食", "ground_truth_keywords": ["素食","纯素","不含肉","蔬食","素餐"]},
        {"type": "forgetting", "question": "用户的宠物叫什么、什么品种？",
         "ground_truth": "豆豆，柴犬", "ground_truth_keywords": ["豆豆","柴犬"]},
    ]))

scripts.append(script("S04", "persona",
    turns=[U("我叫陈昊，深圳一家初创公司的产品经理。"), A("陈昊你好。"),
        U("我有腰椎间盘突出，不能久坐也不能搬重物。"), A("身体情况记下了。"),
        U("我下周要搬家，从南山搬到福田。"), A("理解。"),
        U("给我一份详细的搬家准备清单，注意我的身体情况。")],
    probes=[
        {"type": "consistency", "question": "清单是否考虑不能搬重物？",
         "ground_truth": "不搬重物", "ground_truth_keywords": ["搬家公司","他人协助","请人帮忙","专业搬家","不要自己搬重","不亲自搬","重物"]},
        {"type": "forgetting", "question": "用户从哪里搬到哪里？",
         "ground_truth": "南山到福田", "ground_truth_keywords": ["南山","福田"]},
    ]))

scripts.append(script("S05", "persona",
    turns=[U("我是李子墨，一名独立插画师，自由职业。"), A("插画师。"),
        U("我对花生严重过敏，可能危及生命。"), A("花生过敏。"),
        U("我最近想报一个线下烘焙课程。"), A("好的。"),
        U("帮我筛选课程时要注意什么？要考虑我的健康情况。")],
    probes=[
        {"type": "consistency", "question": "是否强调花生/坚果过敏？",
         "ground_truth": "花生过敏", "ground_truth_keywords": ["花生过敏","花生","告知老师","坚果"]},
        {"type": "forgetting", "question": "用户的职业是什么？",
         "ground_truth": "独立插画师", "ground_truth_keywords": ["插画师","自由职业"]},
    ]))

scripts.append(script("S06", "persona",
    turns=[U("我叫赵明远，刚从英国硕士毕业回国。"), A("海归。"),
        U("我学的是机械工程方向。"), A("机械工程。"),
        U("我现在在准备国内的春招面试。"), A("好的。"),
        U("给我推荐几个适合我背景的目标公司。")],
    probes=[
        {"type": "consistency", "question": "公司是否对应机械工程？",
         "ground_truth": "机械", "ground_truth_keywords": ["机械","制造","车企","重工"]},
        {"type": "forgetting", "question": "用户硕士在哪里读的？",
         "ground_truth": "英国", "ground_truth_keywords": ["英国"]},
    ]))

scripts.append(script("S07", "persona",
    turns=[U("我叫孙悦，今年32岁，二孩妈妈。"), A("孙悦，两个孩子。"),
        U("大孩子8岁男孩，小的3岁女孩。"), A("8岁哥哥，3岁妹妹。"),
        U("我们一家计划春节出国玩。"), A("好。"),
        U("推荐适合我们家庭的目的地，列出三个，注意小孩年龄。")],
    probes=[
        {"type": "consistency", "question": "推荐是否考虑3岁幼儿？",
         "ground_truth": "适合幼儿", "ground_truth_keywords": ["3岁","适合幼儿","幼儿","亲子","低龄","儿童友好"]},
        {"type": "forgetting", "question": "用户家有几个孩子，分别多大？",
         "ground_truth": "两个孩子，8岁和3岁", "ground_truth_keywords": ["8岁","3岁"]},
    ]))

scripts.append(script("S08", "persona",
    turns=[U("我叫周子轩，是个高三复读生。"), A("复读生。"),
        U("今年目标考北京的985院校。"), A("北京985。"),
        U("我数学和英语弱一些，理综还可以。"), A("数学英语弱项。"),
        U("给我一份未来三个月的学习时间分配建议，针对我的弱项。")],
    probes=[
        {"type": "consistency", "question": "是否给数学英语更多权重？",
         "ground_truth": "数学英语", "ground_truth_keywords": ["数学","英语"]},
        {"type": "forgetting", "question": "用户目标院校在哪个城市？",
         "ground_truth": "北京", "ground_truth_keywords": ["北京"]},
    ]))

# recall (8) - long scripts >10 turns
def recall_script(sid, fact, ack, q, gt, kws, themes):
    turns = [U(fact), A(ack)]
    for t in themes: turns.extend(pad(t))
    turns.append(U(q))
    return script(sid, "recall", turns, [
        {"type": "forgetting", "question": q, "ground_truth": gt, "ground_truth_keywords": kws}])

scripts.append(recall_script("S09", "我家wifi密码是Lin8520#家。", "收到。",
    "我家wifi密码是什么来着？", "Lin8520#家", ["Lin8520#家","Lin8520"],
    ["movie","food","stock","tech","music","book"]))
scripts.append(recall_script("S10", "我下周二要去做胃镜，地点是仁济医院东院。", "好的。",
    "我之前提过下周二做胃镜的地点是哪家医院？", "仁济医院东院", ["仁济医院东院","仁济","东院"],
    ["weather","food","movie","music","book","travel"]))
scripts.append(recall_script("S11", "我开的车牌号是沪A·6Q88K。", "好。",
    "我开的车牌号是什么？", "沪A·6Q88K", ["沪A·6Q88K","沪A","6Q88K"],
    ["weather","stock","music","tech","book","travel"]))
scripts.append(recall_script("S12", "我下周三晚上19:30在静安嘉里中心见客户。", "好的。",
    "我之前说过的会议时间和地点是什么？", "下周三19:30，静安嘉里中心",
    ["19:30","静安嘉里中心","下周三"],
    ["weather","food","stock","movie","music","tech"]))
scripts.append(recall_script("S13", "我朋友送了我一只金毛幼犬，3个月大，叫米团。", "米团，金毛。",
    "我之前提过的那只狗叫什么、是什么品种？", "米团，金毛", ["米团","金毛"],
    ["movie","weather","food","music","stock","book"]))
scripts.append(recall_script("S14", "我妈妈的生日是10月17号。", "好的。",
    "我之前提过的妈妈生日是哪天？", "10月17号",
    ["10月17","10/17","10月17号"],
    ["weather","stock","movie","music","tech","travel"]))
scripts.append(recall_script("S15", "我把保险柜密码改成了713942。", "好的。",
    "我刚改的保险柜密码是多少？", "713942", ["713942"],
    ["weather","food","movie","music","book","travel"]))
scripts.append(recall_script("S16", "我老婆的姓名是顾雪薇。", "好的。",
    "帮我写一段送礼物时的卡片，要带上我老婆的名字。", "顾雪薇", ["顾雪薇"],
    ["weather","food","movie","music","book","travel"]))

# update (8)
scripts.append(script("S17", "update",
    turns=[U("我打算下个月去日本旅游。"), A("好。"),
        U("更正一下，行程改了，下个月我要去韩国，不是日本。"), A("好的，韩国。"),
        U("帮我列一份当地必吃美食清单。")],
    probes=[
        {"type": "consistency", "question": "清单是韩国美食？",
         "ground_truth": "韩国", "ground_truth_keywords": ["韩国","韩餐","韩式"]},
        {"type": "forgetting", "question": "用户最终去的国家？",
         "ground_truth": "韩国", "ground_truth_keywords": ["韩国"]},
    ]))
scripts.append(script("S18", "update",
    turns=[U("我家孩子3岁。"), A("好的。"),
        U("更正一下，孩子刚过了4岁生日，应该是4岁不是3岁。"), A("好的，4岁。"),
        U("根据孩子年龄推荐上海的亲子展览。")],
    probes=[
        {"type": "consistency", "question": "推荐是否按4岁？",
         "ground_truth": "4岁", "ground_truth_keywords": ["4岁","四岁"]},
        {"type": "forgetting", "question": "孩子最终年龄？",
         "ground_truth": "4岁", "ground_truth_keywords": ["4岁","四岁"]},
    ]))
scripts.append(script("S19", "update",
    turns=[U("我是个咖啡爱好者，每天大概喝3杯。"), A("好。"),
        U("更正一下，我刚体检医生让减少咖啡因摄入，每天最多1杯了。"), A("好的。"),
        U("基于我现在的咖啡习惯，推荐一款胶囊咖啡机。")],
    probes=[
        {"type": "consistency", "question": "推荐是否基于1杯/天？",
         "ground_truth": "每天1杯", "ground_truth_keywords": ["1杯","一杯"]},
        {"type": "forgetting", "question": "目前每天几杯？",
         "ground_truth": "1杯", "ground_truth_keywords": ["1杯","一杯"]},
    ]))
scripts.append(script("S20", "update",
    turns=[U("我老家在湖南长沙。"), A("好的。"),
        U("更正一下，搞错了，我老家是湖北武汉，不是湖南长沙。"), A("好的。"),
        U("帮我写一段对老家的怀念短文。")],
    probes=[
        {"type": "consistency", "question": "短文提武汉？",
         "ground_truth": "武汉", "ground_truth_keywords": ["武汉","湖北"]},
        {"type": "forgetting", "question": "老家最终是哪？",
         "ground_truth": "湖北武汉", "ground_truth_keywords": ["武汉","湖北"]},
    ]))
scripts.append(script("S21", "update",
    turns=[U("我下周要交的项目截止日是11月20号。"), A("好的。"),
        U("更正一下，截止日改了，现在是11月15号。"), A("好的。"),
        U("基于新截止日，帮我倒推一份本周到截止日的工作计划。")],
    probes=[
        {"type": "consistency", "question": "倒推按11月15？",
         "ground_truth": "11月15号", "ground_truth_keywords": ["11月15","11/15"]},
        {"type": "forgetting", "question": "最终截止日？",
         "ground_truth": "11月15号", "ground_truth_keywords": ["11月15","11/15"]},
    ]))
scripts.append(script("S22", "update",
    turns=[U("我健身房年卡刚续费，2580元。"), A("好的。"),
        U("更正一下，实际扣款是2880不是2580。"), A("好的，2880元。"),
        U("帮我算一下，我每周去3次的话，单次成本多少？")],
    probes=[
        {"type": "consistency", "question": "计算是否基于2880？",
         "ground_truth": "2880", "ground_truth_keywords": ["2880"]},
        {"type": "forgetting", "question": "年卡实际花费？",
         "ground_truth": "2880", "ground_truth_keywords": ["2880"]},
    ]))
scripts.append(script("S23", "update",
    turns=[U("我下个月会议要订10个人的位置。"), A("好的，10人。"),
        U("更正一下，会议位置改成12个人。"), A("好的，12人。"),
        U("基于人数帮我估个人均预算，总预算大概3000。")],
    probes=[
        {"type": "consistency", "question": "按12人算？",
         "ground_truth": "12人", "ground_truth_keywords": ["12人","12个"]},
        {"type": "forgetting", "question": "最终订几个人？",
         "ground_truth": "12人", "ground_truth_keywords": ["12人","12个"]},
    ]))
scripts.append(script("S24", "update",
    turns=[U("我女儿明年要上小学一年级。"), A("好的。"),
        U("更正一下，女儿明年上的是幼儿园中班。"), A("好的。"),
        U("基于她这个阶段，给我推荐合适的绘本。")],
    probes=[
        {"type": "consistency", "question": "绘本对应中班？",
         "ground_truth": "中班", "ground_truth_keywords": ["中班","幼儿园"]},
        {"type": "forgetting", "question": "女儿明年上哪个年级？",
         "ground_truth": "中班", "ground_truth_keywords": ["中班"]},
    ]))

# contradiction (8)
scripts.append(script("S25", "contradiction",
    turns=[U("我的早餐都吃燕麦粥。"), A("好。"),
        U("其实我早餐一般都吃手抓饼配豆浆。"), A("好的。"),
        U("根据我的早餐习惯推荐一份本周菜单。")],
    probes=[{"type": "consistency", "question": "以最新为准？",
        "ground_truth": "手抓饼/豆浆", "ground_truth_keywords": ["手抓饼","豆浆"]}]))
scripts.append(script("S26", "contradiction",
    turns=[U("我对海鲜过敏。"), A("好的。"),
        U("更正下，我之前说错了，是对牛奶过敏，不是海鲜。"), A("好的。"),
        U("帮我安排周末和朋友的聚餐选店，注意我的过敏。")],
    probes=[{"type": "consistency", "question": "以牛奶为准？",
        "ground_truth": "牛奶", "ground_truth_keywords": ["牛奶","乳制品"]}]))
scripts.append(script("S27", "contradiction",
    turns=[U("我家月收入大概4万。"), A("好。"),
        U("其实我家月收入也就1万出头，刚才说错了。"), A("好的。"),
        U("基于我家收入水平给个理财建议。")],
    probes=[{"type": "consistency", "question": "按1万？",
        "ground_truth": "1万", "ground_truth_keywords": ["1万","一万"]}]))
scripts.append(script("S28", "contradiction",
    turns=[U("我女朋友是上海人。"), A("好的。"),
        U("她其实是北京人，我刚记错了。"), A("好的，北京。"),
        U("帮我写一段七夕的告白短信，体现对她家乡文化的了解。")],
    probes=[
        {"type": "consistency", "question": "按北京？",
         "ground_truth": "北京", "ground_truth_keywords": ["北京"]},
        {"type": "forgetting", "question": "女朋友家乡？",
         "ground_truth": "北京", "ground_truth_keywords": ["北京"]},
    ]))
scripts.append(script("S29", "contradiction",
    turns=[U("我体重78公斤。"), A("好。"),
        U("话说回来，我体重68公斤才对，不是78。"), A("好的。"),
        U("根据我的体重，给我一份训练计划。")],
    probes=[{"type": "consistency", "question": "按68？",
        "ground_truth": "68", "ground_truth_keywords": ["68"]}]))
scripts.append(script("S30", "contradiction",
    turns=[U("我是研究生在读。"), A("好的。"),
        U("我口误了，我已经工作三年了，不是在读。"), A("好的。"),
        U("基于我目前身份给我一份职业规划建议。")],
    probes=[{"type": "consistency", "question": "按工作三年？",
        "ground_truth": "工作三年", "ground_truth_keywords": ["工作三年","三年","在职"]}]))
scripts.append(script("S31", "contradiction",
    turns=[U("我有两个孩子。"), A("好的。"),
        U("其实我只有一个孩子，刚才口误了。"), A("好的。"),
        U("基于我的家庭情况，推荐春节出游目的地。")],
    probes=[
        {"type": "consistency", "question": "按一个孩子？",
         "ground_truth": "一个孩子", "ground_truth_keywords": ["一个孩子","三口","一孩"]},
        {"type": "forgetting", "question": "几个孩子？",
         "ground_truth": "一个", "ground_truth_keywords": ["一个","1个"]},
    ]))
scripts.append(script("S32", "contradiction",
    turns=[U("我最近开始喝啤酒。"), A("好。"),
        U("更正下，我从不喝酒，刚才说错了。"), A("好的。"),
        U("基于我的习惯，推荐一家适合我聚会的饭店。")],
    probes=[{"type": "consistency", "question": "按不喝酒？",
        "ground_truth": "不喝酒", "ground_truth_keywords": ["不喝酒","无酒","不饮酒","不含酒"]}]))

for s in scripts:
    last_turn = len(s["turns"])
    for p in s["probes"]:
        p["at_turn"] = last_turn

with open(OUT, "w", encoding="utf-8") as f:
    for s in scripts:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

from collections import Counter
c = Counter(s["scenario"] for s in scripts)
n_p = sum(len(s["probes"]) for s in scripts)
print(f"Wrote {len(scripts)} scripts, {n_p} probes, by scenario: {dict(c)}")
