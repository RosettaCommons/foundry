import os
import numpy as np

def get_lddt(fn):
    data = np.load(fn)
    return data['lddt'].mean()
    #lddt = list()
    #with open(fn) as fp:
    #    for line in fp:
    #        if not line.startswith("ATOM"):
    #            continue
    #        if line[12:16].strip() == "CA":
    #            lddt.append(float(line[61:66]))
    #return np.mean(lddt)

ans_home = "/projects/casp/CASP14/eval/official/answers/domainwise"
dom_s = [line.split() for line in open("/projects/casp/CASP14/eval/official/difficulty.domains")]
if not os.path.exists("model1"):
    os.mkdir("model1")

print ("#DomID diff TM TS LDDT")
for domID, diff in dom_s:
    tar = domID.split('-')[0]
    if not os.path.exists("%s/%s_00_init.pdb"%(tar, tar)):
        continue
    TM_s = list()
    TS_s = list()
    lddt_s = list()
    pdb_s = list()
    for i_iter in range(5):
        pdb_fn = "%s/%s_%02d_init.pdb"%(tar, tar, i_iter)
        if not os.path.exists(pdb_fn):
            continue
        npz_fn = "%s/%s_%02d.npz"%(tar, tar, i_iter)
        lddt = get_lddt(npz_fn)
        lddt_s.append(lddt)
        pdb_s.append(pdb_fn)
    max_idx = np.argmax(lddt_s)
    pdb_fn = pdb_s[max_idx]
    lines = os.popen("TMscore %s %s/%s.pdb"%(pdb_fn, ans_home, domID)).readlines()
    TM = 0.0
    TS = 0.0
    for line in lines:
        if line.startswith("TM-score"):
            TM = float(line.split()[2])
        elif line.startswith("GDT-TS"):
            TS = float(line.split()[1])
            break
    lddt = os.popen("lddt %s %s/%s.pdb | grep Glob"%(pdb_fn, ans_home, domID)).readlines()[-1].split()[-1]
    print (domID, diff, TM, TS, lddt)
    os.system("cp %s model1/%s_pred.pdb"%(pdb_fn, tar))
