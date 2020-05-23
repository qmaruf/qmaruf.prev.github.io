---
title: How To Mount Device From Terminal
date: 2019-01-04 00:00:00 Z
---

Find device list using `lsblk`

Here is the ouput from my ubuntu 16.04 machine.

```
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
sda           8:0    0   1.8T  0 disk 
├─sda2        8:2    0   1.8T  0 part /media/quazi/DATADRIVE1
└─sda1        8:1    0   128M  0 part 
nvme0n1     259:0    0   477G  0 disk 
├─nvme0n1p5 259:5    0 206.1G  0 part /
├─nvme0n1p3 259:3    0   128M  0 part 
├─nvme0n1p1 259:1    0   300M  0 part 
├─nvme0n1p6 259:6    0  31.9G  0 part [SWAP]
├─nvme0n1p4 259:4    0   238G  0 part /media/quazi/Windows
└─nvme0n1p2 259:2    0   500M  0 part /boot/efi
```

If you want to mount the `/sda2` drive, use the following command.
```
udisksctl mount -b /dev/sda2
```
