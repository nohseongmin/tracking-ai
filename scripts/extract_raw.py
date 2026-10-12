import argparse
import json
import logging
import lzma
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(message)s')
with lzma.open(args.source, 'rt', encoding='utf-8') as stream:
    data = json.load(stream)
round_data = next((item for item in data['gameRounds'] if item.get('frames')), None)
if round_data is None:
    raise ValueError('The source has no round with recorded frames')
first_player = next((player for side in ('t', 'ct')
                     for player in (round_data['frames'][0].get(side) or {}).get('players', [])), None)
if first_player is None:
    raise ValueError('The first recorded frame has no player')
records = []
fields = []
for frame_number, frame in enumerate(round_data['frames'], 1):
    for side in ('t', 'ct'):
        for player in (frame.get(side) or {}).get('players', []):
            if player.get('steamID') != first_player['steamID']:
                continue
            for key in player:
                if key not in fields:
                    fields.append(key)
            # Preserve identifiers before JavaScript and Excel numeric precision can change them.
            exported_player = {
                key: str(value) if key == 'steamID'
                else json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                if isinstance(value, (list, dict)) else value
                for key, value in player.items()
            }
            records.append({'frame_number': frame_number, 'tick': frame['tick'], 'player': exported_player})
if not records or data['tickRate'] <= 0:
    raise ValueError('Player records and a positive tick rate are required')
payload = {
    'source_filename': args.source.name,
    'map': data['mapName'], 'tick_rate': data['tickRate'],
    'parse_rate': data['parserParameters']['parseRate'],
    'round': round_data['roundNum'], 'player_name': first_player['name'],
    'team': first_player['team'], 'steam_id': str(first_player['steamID']),
    'fields': fields, 'records': records,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
logging.info('Extracted %s unfiltered records, %s original player fields; tick rate %s',
             len(records), len(fields), payload['tick_rate'])
