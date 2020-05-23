---
title: How To Deploy Whiteglass Theme
date: 2018-07-21 00:00:00 Z
---

I'm keeping a note for myself about how I've create this site. The first step is to create a repository in github using the exact same format given below.
```
your-github-id.github.io
```
For my case, this is
```
qmaruf.github.io
```
Now clone whiteglass repo from the following command.
```
git clone git@github.com:yous/whiteglass.git
```
Enter into the directory using `cd whiteglass` and run the following command to point this repo to your newly created repo.
```
git remote set-url origin git@github.com:your-github-id/your-github-id.github.io.git
```
For me, this is
```
git remote set-url origin git@github.com:qmaruf/qmaruf.github.io.git
```
What this command basically do is to connect the downloaded repo to your repo. Now open the `Gemfile` file and add `jekyll-whiteglass` gem under `jekyll_plugins` group. I'm providing the relevant portions of my Gemfile.
```
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.6"
  gem "jekyll-whiteglass"
  gem "jekyll-remote-theme"
end
```
Add `jekyll-remote-theme` too. Now open the `_config.yml` file and add the following line.
```
remote_theme: yous/whiteglass
```
Update 
```
baseurl: "/whiteglass" # the subpath of your site, e.g. /blog
```
and set 
```
baseurl: "" # the subpath of your site, e.g. /blog
```
Now, to check the site locally, run the following commands.
```
bundle install
bundle exec jekyll serve
```
If everything is Ok you should see your site up and running in this address `localhost:4000`.
Now is the time for you to push and inform everyone about your new home. Commit all changes and push your site to master branch.
```
git commit Gemfile _config.yml -m "remote theme added"
git push -u origin master
```
After this, withing a minute you should have a new home in the following address.
```
your-github-id.github.io
```
