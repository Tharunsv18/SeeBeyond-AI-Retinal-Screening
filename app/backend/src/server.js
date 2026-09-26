const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
require("dotenv").config();

const app = express();

const PORT = process.env.PORT || 5000;

const PROJECT_ROOT = path.resolve(__dirname, "../../..");

const UPLOAD_DIR = path.join(PROJECT_ROOT, "app", "backend", "uploads");
const RESULT_DIR = path.join(PROJECT_ROOT, "app", "backend", "results");

const LAUNCHER = path.join(
  PROJECT_ROOT,
  "deployment",
  "package",
  "run_seeBeyondCLI_fixed.sh"
);

const MATLAB_ROOT = process.env.MATLAB_ROOT || "/usr/local/MATLAB/R2026a";

fs.mkdirSync(UPLOAD_DIR, { recursive: true });
fs.mkdirSync(RESULT_DIR, { recursive: true });

const allowedMimeTypes = new Set([
  "image/jpeg",
  "image/png",
  "image/jpg"
]);

const upload = multer({
  dest: UPLOAD_DIR,
  limits: {
    fileSize: 20 * 1024 * 1024
  },
  fileFilter: (req, file, cb) => {
    if (!allowedMimeTypes.has(file.mimetype)) {
      return cb(new Error("Only JPG and PNG retinal images are allowed."));
    }

    cb(null, true);
  }
});

app.use(cors());
app.use(express.json());

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    service: "SeeBeyond Backend",
    status: "running"
  });
});

app.post("/api/analyze", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      error: "No retinal image uploaded. Use form field: image"
    });
  }

  const inputImage = req.file.path;
  const resultFile = path.join(
    RESULT_DIR,
    `result-${Date.now()}-${process.pid}.json`
  );

  console.log("\n=============================================");
  console.log("SEE BEYOND API ANALYSIS");
  console.log("=============================================");
  console.log(`Input image : ${inputImage}`);
  console.log(`Result file : ${resultFile}`);

  const child = spawn(
    LAUNCHER,
    [
      MATLAB_ROOT,
      inputImage,
      resultFile
    ],
    {
      cwd: PROJECT_ROOT,
      stdio: ["ignore", "pipe", "pipe"]
    }
  );

  let stdout = "";
  let stderr = "";

  child.stdout.on("data", (data) => {
    const text = data.toString();
    stdout += text;
    process.stdout.write(text);
  });

  child.stderr.on("data", (data) => {
    const text = data.toString();
    stderr += text;
    process.stderr.write(text);
  });

  child.on("error", (error) => {
    cleanup(inputImage, resultFile);

    console.error("Failed to start SeeBeyond:", error);

    return res.status(500).json({
      success: false,
      error: "Could not start the SeeBeyond inference engine.",
      details: error.message
    });
  });

  let responseSent = false;

  const sendResult = () => {
    if (responseSent) {
      return true;
    }

    if (!fs.existsSync(resultFile)) {
      return false;
    }

    try {
      const resultText = fs.readFileSync(resultFile, "utf8");
      const result = JSON.parse(resultText);

      responseSent = true;

      console.log("SeeBeyond JSON result detected.");
      console.log("Sending result to client.");

      clearInterval(resultWatcher);
      clearTimeout(resultTimeout);

      cleanup(inputImage, resultFile);

      res.json(result);

      return true;
    } catch (error) {
      return false;
    }
  };

  const resultWatcher = setInterval(() => {
    sendResult();
  }, 500);

  const resultTimeout = setTimeout(() => {
    if (responseSent) {
      return;
    }

    responseSent = true;

    clearInterval(resultWatcher);

    cleanup(inputImage, resultFile);

    return res.status(504).json({
      success: false,
      error: "SeeBeyond inference timed out while waiting for the result.",
      stderr: stderr.slice(-4000),
      stdout: stdout.slice(-4000)
    });
  }, 10 * 60 * 1000);

  child.on("close", (code) => {
    console.log(`SeeBeyond process exited with code ${code}`);

    if (responseSent) {
      return;
    }

    if (code !== 0) {
      responseSent = true;

      clearInterval(resultWatcher);
      clearTimeout(resultTimeout);

      cleanup(inputImage, resultFile);

      return res.status(500).json({
        success: false,
        error: "SeeBeyond inference failed.",
        exitCode: code,
        stderr: stderr.slice(-4000),
        stdout: stdout.slice(-4000)
      });
    }

    // The compiled MATLAB application can finish its work just before
    // the filesystem reports the result JSON. Give the result a few
    // seconds to become readable before returning an error.
    let attempts = 0;
    const postCloseWatcher = setInterval(() => {
      attempts++;

      if (sendResult()) {
        clearInterval(postCloseWatcher);
        return;
      }

      if (attempts >= 20) {
        clearInterval(postCloseWatcher);

        if (responseSent) {
          return;
        }

        responseSent = true;
        clearInterval(resultWatcher);
        clearTimeout(resultTimeout);

        cleanup(inputImage, resultFile);

        return res.status(500).json({
          success: false,
          error: "SeeBeyond completed but did not produce a readable JSON result.",
          exitCode: code,
          stderr: stderr.slice(-4000),
          stdout: stdout.slice(-4000)
        });
      }
    }, 500);
  });
});

app.use((error, req, res, next) => {
  console.error("API error:", error);

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      success: false,
      error: error.message
    });
  }

  return res.status(400).json({
    success: false,
    error: error.message || "Request failed."
  });
});

function cleanup(...files) {
  for (const file of files) {
    try {
      if (file && fs.existsSync(file)) {
        fs.unlinkSync(file);
      }
    } catch (error) {
      console.error(`Cleanup failed for ${file}:`, error.message);
    }
  }
}

app.listen(PORT, () => {
  console.log(`SeeBeyond backend running on http://localhost:${PORT}`);
  console.log(`Analysis endpoint: POST http://localhost:${PORT}/api/analyze`);
});
