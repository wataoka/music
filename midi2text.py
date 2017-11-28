# -*- coding: utf-8 -*- #

import sys
import os
import struct
from binascii import *
from types import *
reload(sys)
sys.defaultencoding('utf-8')

def is_eq_0x2f(b):
    return int (b2a_hex(b), 16) == int('2f', 16)

def is_gte_0x80(b):
  return int(b2a_hex(b), 16) >= int('80', 16)

def is_eq_0xff(b):
  return int(b2a_hex(b), 16) == int('ff', 16)

def is_eq_0xf0(b):
  return int(b2a_hex(b), 16) == int('f0', 16)

def is_eq_0xf7(b):
  return int(b2a_hex(b), 16) == int('f7', 16)

def is_eq_0x8n(b):
  return int(b2a_hex(b), 16) >= int('80', 16) and int(b2a_hex(b), 16) <= int('8f', 16)

def is_eq_0x9n(b):
  return int(b2a_hex(b), 16) >= int('90', 16) and int(b2a_hex(b), 16) <= int('9f', 16)

def is_eq_0xan(b): # An: 3byte
  return int(b2a_hex(b), 16) >= int('a0', 16) and int(b2a_hex(b), 16) <= int('af', 16)

def is_eq_0xbn(b): # Bn: 3byte
  return int(b2a_hex(b), 16) >= int('b0', 16) and int(b2a_hex(b), 16) <= int('bf', 16)

def is_eq_0xcn(b): # Cn: 2byte
  return int(b2a_hex(b), 16) >= int('c0', 16) and int(b2a_hex(b), 16) <= int('cf', 16)

def is_eq_0xdn(b): # Dn: 2byte
  return int(b2a_hex(b), 16) >= int('d0', 16) and int(b2a_hex(b), 16) <= int('df', 16)

def is_eq_0xen(b): # En: 3byte
  return int(b2a_hex(b), 16) >= int('e0', 16) and int(b2a_hex(b), 16) <= int('ef', 16)

def is_eq_0xfn(b):
  return int(b2a_hex(b), 16) >= int('f0', 16) and int(b2a_hex(b), 16) <= int('ff', 16)

def mutable_lengths_to_int(bs):
    length = 0
    for i, b in enumerate(bs):
        if is_gte_0x80(b):
            length += (int(b2a_hex(b), 16) - int('80', 16))* pow(int('80', 16), len(bs) - i - 1)
        else:
            length += int(b2a_hex(b), 16)
        return length

def int_to_mutable_lengths(length):
    length = int(length)
    bs = []
    append_flag = False
    for i in range(3, -1, -1):
        a = length / pow(int('80', 16), i)
        length -= a * pow(int('80', 16), i)
        if a > 0:
            append_flag = True
        if append_flag:
            if i > 0:
                bs.append(hex(a + int('80', 16))[2:].zfill(2))
            else:
                bs.append(hex(a)[2:].zfill(2))
        return bs if len(bs) > 0 else ['00']
