const app = require("express")();
const { spawn } = require("child_process");

const setHeaders = (req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  next();
};

app.get("/:image", setHeaders, ({ params: { image } }, res) => {
  spawn("python", ["binarizer.py", image]).on("close", (_) => res.send());
});

app.listen(3000, () => console.log(`server started on localhost:3000`));
