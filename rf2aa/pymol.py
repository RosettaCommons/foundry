import sys
import os
from icecream import ic

import xmlrpc.client as xmlrpclib

class XMLRPCWrapperProxy(object):
    def __init__(self, wrapped=None):
        self.name = 'cmd'
        self.wrapped = wrapped

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)
        # ic(type(self), attr)
        wrapped = type(self)(attr)
        wrapped.name = name
        return wrapped

    def __call__(self, *args, **kw):
        try:
            return self.wrapped(*args, **kw)
        except Exception as e:
            all_args = tuple(map(str, args))
            all_args += tuple(f'{k}={v}' for k,v in kw.items())
            raise Exception(f"cmd.{self.name}('{','.join(all_args)})'") from e

def get_cmd(pymol_url='http://localhost:9123'):
    cmd = xmlrpclib.ServerProxy(pymol_url)
    if not  ('ipd' in pymol_url or 'localhost' in pymol_url):
        make_network_cmd(cmd)
    return cmd

cmd = None
def init(pymol_url='http://localhost:9123'):
    global cmd
    cmd_inner = get_cmd(pymol_url)
    if cmd is None:
        cmd = XMLRPCWrapperProxy(cmd_inner)
    else:
        cmd.wrapped = cmd_inner
    

def make_network_cmd(cmd):
    # old_load = cmd.load
    def new_load(*args, **kwargs):
        path = args[0]
        with open(path) as f:
            contents = f.read()
        # args[0] = contents
        args = (contents,) + args[1:]
        #print('writing contents')
        cmd.read_pdbstr(*args, **kwargs)
    cmd.is_network = True
    cmd.load = new_load

init()
