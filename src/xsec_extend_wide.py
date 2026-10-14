# Part of qqq-microstructure.
#
# xsec_extend.py, pointed at the WIDE panel. The HuggingFace minute source
# stops at 2026-03, the frozen model's cutoff, so the wide panel has no
# forward month at all. xsec_extend rebuilds missing months from Yahoo daily
# (and 60m/15m) bars in the extractor's schema; this runs the same code --
# imported, not copied, not edited -- with the wide panel's directory and
# universe size: each new month's top 1000 by dollar volume, chosen from the
# names of the last 12 HF months.
#
# Validate first, the house way: --check rebuilds a month the HF wide panel
# already has and prints the universe overlap and price differences. Smaller
# names are where Yahoo is weakest (renames, class shares, thin history), so
# read the check before trusting the forward months.
#
#   python src/xsec_extend_wide.py --check 2026-03    # grade Yahoo, no writes
#   python src/xsec_extend_wide.py                    # build missing months

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xsec_extend as xe

WIDE = os.path.join(xe.ROOT, 'data', 'xsec1000')


def main():
    xe.XSEC = WIDE
    xe.TOP_N = 1000
    if not xe.month_files():
        raise SystemExit('no wide panel in data/xsec1000 -- run '
                         'xsec_extract_fast.py --top 1000 first')
    xe.main()


if __name__ == '__main__':
    main()
