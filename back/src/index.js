const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const ext = path.extname(file.originalname) || '.mp4';
    const safeName = `upload-${timestamp}${ext}`;
    cb(null, safeName);
  }
});


const upload = multer({ storage });


app.get("/tch", (req,res) =>{
  const htmlPath = path.join(__dirname, "../../front/tch/tch-index.html");
  res.sendFile(htmlPath);
});

// app.post("/uploads", upload.single("video"), (req, res) => {
//   if (!req.file) return res.status(400).json({ error: "動画ファイルが必要です" });

//   const videoPath = req.file.path;
//   const videoName = path.parse(req.file.originalname).name;
//   const scriptPath = path.join(__dirname, "../scripts/process_video.py");
//   const pythonPath = path.join(__dirname, "../../venv/bin/python");


//   execFile(pythonPath, [scriptPath, videoPath, videoName], (err, stdout, stderr) => {
//     fs.unlinkSync(videoPath);
//     if (err) {
//       console.error(stderr);
//       return res.status(500).json({ error: "動画処理失敗", stderr });
//     }
//     res.json({ message: "処理完了", output: stdout.trim() });
//   });
// });


app.post("/uploads", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "動画ファイルが必要です" });

  const videoPath = req.file.path;
  const { course, section, title, videoId } = req.body;
  const videoName = title || path.parse(req.file.originalname).name;
  const scriptPath = path.join(__dirname, "../scripts/process_video.py");
  
  //ocr無しで音声認識をする場合　比較用
  //const scriptPath = path.join(__dirname, "../scripts/asr_no_prompt.py");
  
  const pythonPath = path.join(__dirname, "../../venv/bin/python");

  console.log("受け取ったメタ情報:");
  console.log("コース:", course);
  console.log("セクション:", section);
  console.log("動画名:", title);
  console.log("動画ID:", videoId);

  // Pythonスクリプトに必要ならメタ情報を渡す
  const args = [scriptPath, videoPath, videoName, course, section, videoId];

  execFile(pythonPath, args, (err, stdout, stderr) => {
    fs.unlinkSync(videoPath); // 一時ファイル削除
    if (err) {
      console.error(stderr);
      return res.status(500).json({ error: "動画処理失敗", stderr });
    }
    res.json({ message: "処理完了", output: stdout.trim() });
  });
});

app.listen(3001, () => {
  console.log("Server running at http://localhost:3001");
});

