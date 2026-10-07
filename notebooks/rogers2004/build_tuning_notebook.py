"""Create a repository notebook for viewing and checking the tuned result."""
from pathlib import Path
import base64
import nbformat as nb

root=Path(__file__).resolve().parents[2]
output=root/'outputs/rogers2004/tuning'
notebook=nb.v4.new_notebook()
notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
notebook.cells=[nb.v4.new_markdown_cell('# Rogers (2004)：调参与稳定性验证\n\n此笔记本在项目目录内运行。原始实验未覆盖。训练目标、数据修订、失败尝试和独立损伤验证都保存在调参报告中；这是优化扩展，不是原论文全部参数的严格复现。'),
nb.v4.new_code_cell('from pathlib import Path\nimport sys, json, torch\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "notebooks/rogers2004/rogers_model.py").exists())\nsys.path.insert(0, str(root / "notebooks/rogers2004"))\nimport rogers_model as m\nfrom tune_model import audit\ntorch.set_num_threads(1)\noutput = root / "outputs/rogers2004/tuning"\nselection = json.loads((output / "selection.json").read_text())\nmodel, environment = m.load(output / selection["variant"])\nwith torch.no_grad():\n    print(json.dumps(audit(model, environment), indent=2))'),
nb.v4.new_markdown_cell((output/'REPORT.md').read_text(encoding='utf8').replace('![训练比较](training_comparison.png)','').replace('![损伤验证](lesion_validation.png)',''))]
for name in ['training_comparison.png','lesion_validation.png']:
    cell=nb.v4.new_markdown_cell(f'![{name}](attachment:{name})')
    cell.attachments={name:{'image/png':base64.b64encode((output/name).read_bytes()).decode()}}
    notebook.cells.append(cell)
notebook.cells.append(nb.v4.new_markdown_cell('训练入口：`tune_model.py`、`refine_stability.py`、`refine_adam.py`。验证入口：`validate_tuning.py`。具体命令和假设见 `notes/ROGERS2004_NOTES.md`。图中误差条来自一个模型的随机损伤掩码，不代表患者差异或初始化差异。'))
nb.write(notebook,Path(__file__).with_name('tuning_results.ipynb'))
