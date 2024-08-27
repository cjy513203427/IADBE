# 📦 Full Installation

Create a new folder
```shell
mkdir IADBE_ALL
cd IADBE_ALL
```

Clone project
```shell
# clone IADBE core
git clone https://github.com/cjy513203427/IADBE.git
# clone IADBE server
git clone https://github.com/cjy513203427/IADBE_Server.git
# clone IADBE frontend
git clone https://github.com/cjy513203427/IADBE_Frontend.git
```

Copy docker-compose.yml here.
```shell
cp IADBE/docker-compose.yml ./
```

So, the file structure should look like this:
```shell
├── IADBE
│   ├── docker-compose.yml
│   ├── IADBE
│   ├── IADBE_Frontend
│   └── IADBE_Server

```

Build docker images with docker-compose.yml
```shell
docker-compose up --build -d
```
Till now you should be able to access IADBE system.
1. IADBE Server: http://localhost:8080/api/xxx
2. IADBE Frontend: http://localhost:4200/
3. IADBE: Set as a dev env (Recommended) or train directly in container.
<br/><br/>

Stop running container
```shell
docker-compose down
```
