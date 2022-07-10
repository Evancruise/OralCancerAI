# Running a Container
Running of containers is managed with the Docker run command. To run a container in an interactive mode, first launch the Docker container.

```bash
docker run -p 8000:8000 -t -i oralcancer001
```
-p: port (from 8000 to 8000 in this bash line)
-t: run on terminal
-i: iteraction mode

Then hit Crtl+p and you will return to your OS shell.

