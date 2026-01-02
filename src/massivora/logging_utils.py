import logging
import os


def setup_logging(cfg):
    log_cfg = cfg.get('logging', {})
    if log_cfg and log_cfg.get('enable'):
        log_file = log_cfg.get('file') or 'errors.log'
        level_name = str(log_cfg.get('level') or 'WARNING').upper()
        try:
            level = getattr(logging, level_name, logging.WARNING)
        except AttributeError:
            level = logging.WARNING
        log_dir = cfg.get('logging', {}).get('dir')
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, log_file)

        # Force reconfiguration so we can reliably control when logging is set up
        logging.basicConfig(
            filename=log_file,
            level=level,
            format='%(asctime)s %(levelname)s - From %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True,
        )
