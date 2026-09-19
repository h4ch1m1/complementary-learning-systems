"""Display helpers; model and training remain in the notebook."""
import html
import base64
import json
import re
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from IPython.display import HTML, display


class TrainingView:
    def __init__(self, labels, indices):
        self.labels, self.indices = labels, indices
        self.frames = []
        directory = Path(__file__).resolve().parents[2] / 'assets' / 'photos'
        credits = json.loads((directory / 'credits.json').read_text(encoding='utf-8'))
        self.photos = {}
        attribution = []
        for label in labels:
            credit = credits[label]
            image = directory / credit['file']
            mime = 'image/png' if image.suffix == '.png' else 'image/jpeg'
            self.photos[label] = f'data:{mime};base64,' + base64.b64encode(image.read_bytes()).decode()
            meta = credit['metadata']
            clean = lambda key: html.escape(html.unescape(re.sub('<[^>]+>', '', meta.get(key, {}).get('value', ''))))
            attribution.append(f'<li><a href="{html.escape(credit["source"], quote=True)}" target="_blank" rel="noopener">{label}</a> — {clean("Artist")} · {clean("LicenseShortName")}</li>')
        self.credits = '<ul>' + ''.join(attribution) + '</ul>'
        self.handle = display(HTML('Preparing training…'), display_id=True)

    def update(self, snapshots, losses):
        # Only compute newly recorded epochs; dragging the slider needs no kernel.
        for epoch in range(len(self.frames), len(snapshots)):
            activity = snapshots[epoch][self.indices]
            tree = dendrogram(linkage(activity, method='average'), no_plot=True,
                              color_threshold=0.45, above_threshold_color='C0')
            self.frames.append({'activity': activity.round(6).tolist(),
                                'icoord': tree['icoord'], 'dcoord': tree['dcoord'],
                                'order': tree['leaves'], 'colors': tree['color_list'],
                                'leafColors': tree['leaves_color_list']})
        largest = max(max(max(row) for row in f['dcoord']) for f in self.frames)
        data = {'labels': self.labels, 'frames': self.frames, 'losses': losses, 'photos': self.photos,
                'maxDistance': max(1.5, float(np.ceil(largest * 2) / 2))}
        template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
        self.document = template.replace('__TRAINING_DATA__', json.dumps(data)).replace('__PHOTO_CREDITS__', self.credits)
        iframe = '<iframe title="Semantic representation learning" style="width:100%;height:1800px;border:0" srcdoc="{}"></iframe>'
        self.handle.update(HTML(iframe.format(html.escape(self.document, quote=True))))

    def save(self, path='outputs/training_replay.html'):
        Path(path).write_text(self.document, encoding='utf-8')
