# Part of qqq-microstructure. Set QQQ_DBN_ZIP to your Databento archive.
# Streams one day at a time out of the zip: the archive is larger than most free disks.
import os
import databento as db, pandas as pd, numpy as np, sys
f = sys.argv[1]
store = db.DBNStore.from_file(f)
print("metadata:", store.metadata.schema, store.metadata.stype_out, store.metadata.start, store.metadata.end)
df = store.to_df()
print("rows:", len(df))
print("cols:", list(df.columns))
print(df.head(5).to_string())
print("---dtypes---"); print(df.dtypes)
