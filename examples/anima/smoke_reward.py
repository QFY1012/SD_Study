"""DIAGNOSTIC ONLY: image mean brightness, NOT anatomy or human preference.
This exercises reward I/O and backward/update. Never use for a real preference run.
"""
import json
import sys
from PIL import Image, ImageStat
request = json.load(open(sys.argv[1]))
scores = []
for sample in request['samples']:
    with Image.open(sample['image']) as image:
        value = ImageStat.Stat(image.convert('L')).mean[0] / 255
    scores.append({'id': sample['id'], 'reward': value})
with open(sys.argv[2], 'w') as out:
    json.dump({'scores': scores}, out)
