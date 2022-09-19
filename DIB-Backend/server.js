const app = require("express")();
const { spawn } = require("child_process");

const setHeaders = (req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET, POST, OPTIONS, PUT, PATCH, DELETE"
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "X-Requested-With,content-type,authorization"
  );
  res.setHeader("Access-Control-Allow-Credentials", "true");
  next();
};

app.use(setHeaders);

app.get("/:image", (req, res) => {
  // console.log(req.params.image);
  const pyScript = spawn("python", ["binarizer.py", req.params.image]);
  // pyScript.stdout.on("data", (d) => (data = console.log(d.toString())));
  pyScript.on("close", (_) => res.json({ msg: "done!!" }));
});
// let data;

app.listen(3000, () => console.log(`server started on localhost:3000`));
