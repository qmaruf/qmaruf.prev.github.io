It took me 48 hours to install and run Carla in my Ubuntu 16.04 workstation. So, what was the problem? When I was trying to run `python spawn_npc.py -n 80` to create 
80 vehicles in the simulator, it showed an error regarding a missing library -- `libxerces-c-3.2.so`. I did not find any suitable approach to install this library, tried 
to install Carla in docker, but there were more than one issues. I can't remember now.

At last, I ran the following command to find any existing library in my system.

```
locate libxerces
````

It shows the following output.

```
/usr/lib/x86_64-linux-gnu/libxerces-c-3.1.so
/usr/local/MATLAB/R2019b/bin/glnxa64/libxerces-c-3.2.so
/usr/local/MATLAB/R2019b/bin/glnxa64/libxerces-c.so
/usr/local/MATLAB/R2019b/toolbox/sl3d/orbisnap/bin/glnxa64/libxerces-c-3.2.so
/usr/local/MATLAB/R2019b/toolbox/sl3d/orbisnap/bin/glnxa64/libxerces-c.so
/usr/share/doc/libxerces-c3.1
/usr/share/doc/libxerces-c3.1/changelog.Debian.gz
/usr/share/doc/libxerces-c3.1/copyright
/usr/share/lintian/overrides/libxerces-c3.1
/var/lib/dpkg/info/libxerces-c3.1:amd64.list
/var/lib/dpkg/info/libxerces-c3.1:amd64.md5sums
/var/lib/dpkg/info/libxerces-c3.1:amd64.shlibs
/var/lib/dpkg/info/libxerces-c3.1:amd64.triggers
```

Hmm. Somehow, the `libxerces-c-3.2.so` was previously installed along with `MATLAB`! Now the next step is to add the following command in `~/.bashrc` file.

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/MATLAB/R2019b/bin/glnxa64/
```

Well, the problem is solved now. Here is a screen shot of Carla -- up and running.
![Screenshot from 2021-01-19 17-22-50](https://user-images.githubusercontent.com/530250/105001296-46c93200-5a7b-11eb-8144-3b46ac1d2106.png)



