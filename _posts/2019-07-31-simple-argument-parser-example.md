---
title: Simple Argument Parser Example
date: 2019-07-31 00:00:00 Z
---

```python
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-f','--foo', help='Description for foo argument', required=True)
parser.add_argument('-b','--bar', help='Description for bar argument', required=True)
args = vars(parser.parse_args())

print (args)
```
Source https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results
