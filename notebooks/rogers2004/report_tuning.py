"""Report actual controlled runs, without selecting on lesion outcomes."""
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rogers_model as m

def main():
    root=m.ROOT/'outputs/rogers2004/tuning'
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    lines=['# 继续训练与输入修订实验','','原有 `paper_settings` 结果保持不变。三组均从原始400轮权重继续训练；前200轮学习率0.005，后200轮0.002。选择依据是完整网络表现，未使用损伤曲线选参数。','',
           '|实验|新增轮数|命名正确率|特征判断正确率|误差<0.05比例|最大误差|未稳定输入|',
           '|---|---:|---:|---:|---:|---:|---:|']
    labels={'continue':'Original margin .05','tight':'Margin .01','data':'Margin .01 + revised data'}
    if (root/'stable/training.json').exists(): labels['stable']='Extended trajectory + hard targets'
    if (root/'stable_long/training.json').exists(): labels['stable_long']='Higher LR (failed, stopped)'
    if (root/'adam/training.json').exists(): labels['adam']='Full-batch Adam extension'
    for name,label in labels.items():
        rows=json.loads((root/name/'training.json').read_text())
        r=rows[-1]
        lines.append(f'|{name}|{r["additional_epoch"]}|{r["naming"]:.2%}|{r["bit_accuracy"]:.3%}|{r["within_005"]:.2%}|{r["max_error"]:.4f}|{r["unsettled"]}|')
        for ax,key in zip(axes,['mse','within_005','ticks']):
            ax.plot([r['additional_epoch'] for r in rows],[r[key] for r in rows],marker='o',label=label)
            ax.set_xlabel('Stage updates (SGD epochs / Adam steps)',fontsize=9); ax.set_ylabel(key); ax.grid(alpha=.2)
    axes[0].set_yscale('log'); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'training_comparison.png',dpi=180); plt.close(fig)
    lines += ['', '输入修订只改动8个非标签百科特征：取消重复的概率性动物/人工物标签，并补入图3所示水果共享的4个特征；精确类别标签、名称、视觉输入和随机抽样数保持不变。图3与正文存在其他歧义，因此仍是重建数据，不能称为原始数据。', '',
              '容差0.01改变了训练目标，并非论文参数的原样复现。小于0.05的统计排除一般名称的歧义线索；命名与稳定性均采用最长2000步的完整网络评估。数据组在旧数据权重基础上继续学习，属于适应实验，不代表从头训练的独立复现。', '',
        '稳定性扩展组 stable 从 data 组继续：展开56步，对21—56步评分，硬目标0/1与0.01误差死区，学习率0.002。它同时修改了时间展开和目标形式，不能单独归因于其中一项。横轴是各阶段新增轮数；stable 的起点已额外训练400轮。', '',
        'stable_long 提高学习率到0.005后崩溃，提前停止；不是已完成800轮。adam 从 stable 权重开始，使用全批量Adam (lr=0.001)、56步展开、一般名称的平均特征目标和二元交叉熵。这些都是明确的优化扩展；其横轴是全批量更新次数，不是原来的逐样本SGD轮数。', '',
        '![训练比较](training_comparison.png)']
    selected_path=root/'selection.json'
    if selected_path.exists():
        selection=json.loads(selected_path.read_text()); chosen=selection['variant']; output=root/chosen
        lines += ['',f'选用实验：**{chosen}**。选择记录：`selection.json`。']
        if (output/'summary.json').exists():
            summary=json.loads((output/'summary.json').read_text())
            raw=list(csv.DictReader((output/'lesion_trials.csv').open()))
            unsettled=sum(int(r['nonconvergent_inputs'])>0 for r in raw)
            lines += ['', f'独立损伤随机种子1707，每档50个掩码，共451次评估；{unsettled}次包含未稳定输入。全部保留，不因不稳定而删除。', '',
                      '|20%损伤指标|实测均值|','|---|---:|']
            for key in ['naming/all/correct','matching/close','matching/distant','matching/unrelated']:
                if key in summary['0.2']: lines.append(f'|{key}|{summary["0.2"][key]["mean"]:.2%}|')
            fig,axes=plt.subplots(2,3,figsize=(14,8))
            axes=axes.ravel()
            groups=[('Naming',['naming/all/correct','naming/all/semantic','naming/all/omission']),
                    ('Word-picture matching',['matching/close','matching/distant','matching/unrelated']),
                    ('Sorting pictures',['sorting/picture/animal_artifact/general','sorting/picture/animal_artifact/specific']),
                    ('Drawing omissions',['drawing/shared_domain/omission','drawing/shared_category/omission','drawing/distinctive/omission']),
                    ('Fruit sorting',['sorting/picture/fruit/general','sorting/picture/fruit/specific']),
                    ('Dynamics',['nonconvergent_inputs'])]
            x=sorted(float(k) for k in summary)
            for ax,(title,keys) in zip(axes,groups):
                for key in keys:
                    if key not in summary['0.2']: continue
                    y=[summary[str(v)][key]['mean'] for v in x]
                    label=key.split('/')[-2] if title=='Drawing omissions' else key.split('/')[-1]
                    sem=[summary[str(v)][key]['sem'] for v in x]
                    ax.errorbar(x,y,yerr=sem,marker='o',label=label,markersize=3,capsize=2)
                ax.set_title(title); ax.set_xlabel('Fraction of connections removed'); ax.legend(fontsize=8); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(root/'lesion_validation.png',dpi=180); plt.close(fig)
            lines += ['', '![损伤验证](lesion_validation.png)']
    lines += ['', '局限：仅一个初始化和一个环境种子；未拟合患者数据。完整网络表现改善不等于所有损伤状态都稳定，也不等于满足全部原论文数值标准。原始论文与当前曲线没有数字化逐点比较。']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')

if __name__=='__main__': main()
