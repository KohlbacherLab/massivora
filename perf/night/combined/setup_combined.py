"""Stage the 'combined' timeline mini project on the SAME 2,000 screen pairs as night_timeline
(pairs.txt copied from there), variant 'combined' (night/combined = perf/fused-screen + night/hostside +
night/policy-gate).   setup_combined.py
Writes $R/combined/{msa -> helpy_msa, ecs/, logs/, gauss.yml, plm.yml, gauss.sqlite3, plm.sqlite3}.
  gauss: fused (keep_concatenated false), gpu_host_processes 16, shared-source cache on (/dev/shm), no pre-gate
  plm:   cascade keep_fraction 0.20 + min_neffoverL 0.1, uncapped fp16 + maxeval 50 (downsample 30,000),
         gpu_host_processes 16, shared-source cache on
Configs come from the variant's config_template/project.yml (as setup.py of night_timeline)."""
import os, shutil, sqlite3, sys
B = '/scratch/kohlbach/massivora-bench'
SITE = f'{B}/variants/combined'
R = f'{B}/runs/night_combined'
TL = f'{B}/runs/night_timeline'
SS_CACHE = os.environ.get('SS_CACHE', '/dev/shm/kohlbach_ss_k10')
P_HOST = int(os.environ.get('HOST_PROCS', '16'))
sys.path.insert(0, SITE)
import yaml
from massivora.db import create_job_db, STATUS

sample = [l.strip() for l in open(f'{TL}/pairs.txt') if l.strip()]
assert len(sample) == 2000, len(sample)
os.makedirs(R, exist_ok=True)
shutil.copy(f'{TL}/pairs.txt', f'{R}/pairs.txt')
pids = sorted({p for n in sample for p in n.split('-', 1)})
base = yaml.safe_load(open(f'{SITE}/massivora/config_template/project.yml'))

mode = 'combined'
P = f'{R}/{mode}'
shutil.rmtree(P, ignore_errors=True)
os.makedirs(f'{P}/logs'); os.makedirs(f'{P}/ecs')
os.symlink(f'{B}/data/helpy_msa', f'{P}/msa')
open(f'{P}/proteins.txt', 'w').write('\n'.join(pids) + '\n')


def write(stem, db, table, logfile, couple, cascade=None, downsample_to=30000):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg['project'].update(name=stem, project_path=P, proteins_list=f'{P}/proteins.txt')
    cfg['paths'] = {'monomers': 'msa', 'couplings': 'ecs'}
    cfg['database'] = {'backend': 'sqlite', 'filename': db, 'shared_filesystem': False}
    cfg['logging'] = {'enabled': True, 'level': 'INFO', 'dir': f'{P}/logs', 'file': logfile}
    cfg['concatenate']['db_table'] = table
    cfg['concatenate']['downsample'] = {'enabled': True, 'to': downsample_to}
    cfg['concatenate']['shared-source_cache'] = SS_CACHE
    cfg['couple'].update(db_table=table, **couple)
    cfg['couple'].pop('cascade', None)
    if cascade:
        cfg['couple']['cascade'] = cascade
    yaml.safe_dump(cfg, open(f'{P}/{stem}.yml', 'w'), sort_keys=False)
    return cfg


def make_db(cfg, db, table):
    if os.path.exists(db):
        os.remove(db)
    create_job_db(cfg)
    c = sqlite3.connect(db)
    c.executemany('INSERT INTO alignments (pid, status) VALUES (?, ?)', [(p, STATUS['DONE']) for p in pids])
    c.executemany(f'INSERT INTO "{table}" (name, pid1, pid2, status) VALUES (?, ?, ?, ?)',
                  [(n, *n.split('-', 1), STATUS['NOOPT']) for n in sample])
    c.commit(); c.close()


cfg = write('gauss', 'gauss.sqlite3', 'couplings', 'gauss.log',
            dict(method='gauss', gpu_sessions=4, gpu_host_processes=P_HOST, gpu_workers=4 + 2 * P_HOST,
                 keep_concatenated=False))
make_db(cfg, f'{P}/gauss.sqlite3', 'couplings')
cfg = write('plm', 'plm.sqlite3', 'plm', 'plm.log',
            dict(method='plm', maxit=50, fp16_history=True, gpu_host_processes=P_HOST, gpu_workers=4 + 2 * P_HOST,
                 keep_concatenated=False),
            cascade={'source_db': 'gauss.sqlite3', 'source_table': 'couplings',
                     'keep_fraction': 0.20, 'rank_by': 'log_lr_raw', 'min_neffoverL': 0.1},
            downsample_to=30000)
make_db(cfg, f'{P}/plm.sqlite3', 'plm')
print(f'staged {len(sample)} pairs, {len(pids)} proteins under {P} (cache {SS_CACHE}, host procs {P_HOST})')
