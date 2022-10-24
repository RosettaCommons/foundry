import re
from collections import OrderedDict
import pandas as pd

def parse_training_log(filename):
    headers = ['Local','Train','Monomer','Homo','Hetero','NA','NAfs','RNA','SM Compl']
    records = []
    with open(filename) as f:
        for line in f:
            if line.startswith("Header"):
                # renaming a few things from an earlier version of training script
                for src,tgt in [('# epochs','num_epochs'), ('processed','examples_seen_in_epoch'),
                                ('examples in epoch','examples_per_epoch'), ('Max mem','max_mem'),
                                ('seconds','time'), ('total_loss: loss','Total_loss: total_loss')]:
                    line = line.replace(src,tgt)

                columns = re.findall('(\w+)',line)
                for val in ['Batch','Time','Total_loss']:
                    if val in columns: columns.remove(val)

            if any([line.startswith(h) for h in headers]):
                values = [line.split(':')[0]]+[float(x) for x in re.findall('(\d+\.*\d*)',line)]
                records.append(OrderedDict(zip(columns, values)))

    df = pd.DataFrame.from_records(records)
    df = df.drop_duplicates(['Header','epoch','examples_seen_in_epoch'])

    df_s = []
    offset = 0
    for ep in df['epoch'].drop_duplicates():
        tmp = df[df['epoch']==ep]

        n_per_epoch = tmp['examples_per_epoch'].values[0]
        tmp['example'] = tmp['examples_seen_in_epoch']+offset
        offset += n_per_epoch

        mask = tmp['Header']!='Local'
        tmp.loc[mask,'example'] = offset

        df_s.append(tmp)
    df = pd.concat(df_s)

    return df

