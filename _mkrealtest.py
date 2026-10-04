# 把 dataset_demo 的 val 真实帧转成单类测试集 realtest（class1 -> 0, name=机器人）
import shutil
from pathlib import Path

src = Path('dataset_demo')
dst = Path('realtest')
imgs = sorted((src / 'images' / 'val').glob('*.jpg'))
dst_img = dst / 'images' / 'val'
dst_lab = dst / 'labels' / 'val'
dst_img.mkdir(parents=True, exist_ok=True)
dst_lab.mkdir(parents=True, exist_ok=True)

n_box = 0
for p in imgs:
    shutil.copyfile(p, dst_img / p.name)
    lab = (src / 'labels' / 'val' / (p.stem + '.txt')).read_text(encoding='utf-8').strip()
    out = []
    for line in lab.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == '1':      # 只保留 机器人(class1)，重映射到 0
            out.append('0 ' + ' '.join(parts[1:]))
            n_box += 1
    (dst_lab / (p.stem + '.txt')).write_text('\n'.join(out), encoding='utf-8')

(dst / 'data.yaml').write_text(
    '# 真实帧测试集（单类）—— 标签来自你原模型对真实帧的伪标注\n'
    f'path: {dst.resolve().as_posix()}\n'
    'train: images/val\n'
    'val: images/val\n'
    'names:\n  0: 机器人\n',
    encoding='utf-8')
print(f'realtest: {len(imgs)} 张图, {n_box} 个框 -> {dst.resolve()}')
