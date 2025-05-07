# Building asap_run

Edit new Dockerfile.vX
Update Dockerfile symlink
Edit image tag in docker-compose.yml

docker-compose build

### OLD

# asap_run
ASAP container for running all pipeline code

# How to build the latest asap_run Docker from this git clone
```bash
docker build . -t asap_run:v8
```

# How to use pre-built Docker images
All pre-built images for asap-run are available on [DockerHub](https://hub.docker.com/r/fabdavid/asap_run/tags).
You can pull any version by using the pull commmand:
```bash
docker pull fabdavid/asap_run:v7
```

# How to run a command from a container:

- You can open a shell in a Docker, and for example run "Rscript" from within the container
```bash
docker run -it fabdavid/asap_run:v8 bash
```
Then run
```bash
Rscript myscript.R
```

- You can also directly run the command from within the container, but without entering the container per se
```bash
docker run --rm fabdavid/asap_run:v8 Rscript myscript.R
```
Note: Use the -v option to mount any volume on the Docker container, so that you can access the necessary files
  
# How to extract the list of R package versions from a docker
```bash
grep -Ri "^version:" /usr/local/lib/R/library/ | grep DESCRIPTION
```
