git push -u old-origin(github)/origin(git.iouou) --all


Command line instructions

Git global setup
    git config --global user.name "rensijie"
    git config --global user.email "rensijie@ooyby.com"

Create a new repository
    git clone http://git.iouou.cn/rensijie/PPA-classification-prediction.git
    cd empty-test
    touch README.md
    git add README.md
    git commit -m "add README"
    git push -u origin master

Existing folder
    cd existing_folder
    git init
    git remote add origin http://git.iouou.cn/rensijie/PPA-classification-prediction.git
    git add .
    git commit -m "Initial commit"
    git push -u origin master

Existing Git repository
    cd existing_repo
    git remote rename origin old-origin
    git remote add origin http://git.iouou.cn/rensijie/PPA-classification-prediction.git
    git push -u origin --all
    git push -u origin --tags