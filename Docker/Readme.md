# Running a Container

1. Build a container called "oralcancer001"

```bash
docker build -t oralcancer001 .
```

2. Launch the Docker container.

```bash
docker run -p 8000:8000 -t -i oralcancer001
```
-p: port (from 8000 to 8000 in this bash line)
-t: run on terminal
-i: iteraction mode

Then hit Crtl+p and you will return to your OS shell.

You will then be running in the instance of the CentOS system on the Ubuntu server.

3. Runs a new command in a running container.

```bash
docker exec -it {CONTAINER_ID} /bin/sh
```
