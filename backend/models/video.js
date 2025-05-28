const mongoose = require("mongoose");

const VideoSchema = new mongoose.Schema({
  filename: String,
  uploadDate: { type: Date, default: Date.now },
  violenceStatus: String
});

module.exports = mongoose.model("Video", VideoSchema);
