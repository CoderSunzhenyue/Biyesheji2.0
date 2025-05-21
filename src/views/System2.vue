<template>
  <div class="app">
    <el-card class="card">
      <div class="card-header">
        <h2>语音敏感信息检测</h2>
      </div>

      <div class="audio-processing-area">
        <el-upload
          ref="uploadRef"
          v-model:file-list="fileList"
          :action="uploadUrl"
          :auto-upload="false"
          multiple
          :limit="10"
          :accept="allowedFileTypes.join(',')"
          :on-change="handleFileChange"
          :on-remove="handleFileRemove"
          :on-exceed="handleExceed"
          class="upload-demo"
        >
          <el-button type="primary">选择音频文件</el-button>
          <template #tip>
            <div class="el-upload__tip">
              支持上传
              {{ allowedFileTypes.join(', ') }} 格式文件，单个文件大小不超过
              {{ MAX_FILE_SIZE_MB }}MB。
            </div>
          </template>
        </el-upload>

        <div class="action-buttons">
          <el-button
            class="upload-button"
            type="success"
            @click="submitUpload"
            :loading="loading"
            :disabled="fileList.length === 0 || loading"
          >
            开始检测
          </el-button>

          <el-button
            type="info"
            @click="clearResults"
            :disabled="
              (analysisResults.length === 0 && fileList.length === 0) || loading
            "
          >
            清空全部
          </el-button>
        </div>

        <el-divider />

        <div
          class="analysis-results-area"
          v-if="analysisResults.length > 0 || loading"
        >
          <h3>检测结果</h3>
          <div v-if="loading" class="loading-indicator">
            <el-icon class="is-loading"><Loading /></el-icon> 正在处理中...
          </div>

          <div v-else class="results-list">
            <div
              v-for="(audioResult, index) in analysisResults"
              :key="audioResult.filename + '_' + index"
              class="audio-result-item"
            >
              <div class="result-header">
                <strong>文件名:</strong> {{ audioResult.filename }}
                <span
                  :class="{
                    'category-label': true,
                    'category-personal-privacy':
                      audioResult.overall_classification === '个人隐私信息',
                    'category-porn':
                      audioResult.overall_classification === '黄色信息',
                    'category-bad-harmful':
                      audioResult.overall_classification === '不良有害信息',
                    'category-non-sensitive':
                      audioResult.overall_classification === '无敏感信息',
                    'category-failed':
                      // 处理后端返回的各种非敏感分类状态（如：处理失败, 转写失败或无内容, 不支持的音频格式）
                      audioResult.overall_classification !== '个人隐私信息' &&
                      audioResult.overall_classification !== '黄色信息' &&
                      audioResult.overall_classification !== '不良有害信息' &&
                      audioResult.overall_classification !== '无敏感信息'
                  }"
                >
                  {{ audioResult.overall_classification || '处理异常' }}
                </span>
              </div>

              <div
                class="transcribed-text-block"
                v-if="audioResult.transcribed_text"
              >
                <h4>转录文本:</h4>
                <pre>{{ audioResult.transcribed_text }}</pre>
              </div>
              <div
                v-else-if="
                  audioResult.overall_classification === '转写失败或无内容'
                "
              >
                语音转写失败或结果为空。
              </div>
              <div
                v-else-if="
                  audioResult.overall_classification === '不支持的音频格式'
                "
              >
                不支持该音频格式。
              </div>

              <div
                v-if="
                  // 仅在总体分类不是“无敏感信息”且有敏感片段时显示
                  audioResult.overall_classification !== '无敏感信息' &&
                  Array.isArray(audioResult.sensitive_segments) &&
                  audioResult.sensitive_segments.length > 0
                "
              >
                <h4>
                  敏感信息发现 ({{ audioResult.sensitive_segments.length }} 条):
                </h4>
                <ul class="segments-list">
                  <li
                    v-for="(
                      segment, segIndex
                    ) in audioResult.sensitive_segments"
                    :key="segIndex"
                    class="segment-item"
                  >
                    <div class="segment-category">
                      <strong>类别:</strong>
                      <span
                        :class="{
                          'category-label': true,
                          'category-personal-privacy':
                            segment.category === '个人隐私信息',
                          'category-porn': segment.category === '黄色信息',
                          'category-bad-harmful':
                            segment.category === '不良有害信息'
                          // 敏感片段列表中不展示“无敏感信息”的片段
                        }"
                      >
                        {{ segment.category }}
                      </span>
                    </div>
                    <div class="segment-snippet">
                      <strong>片段:</strong>
                      <pre>{{ segment.text }}</pre>
                    </div>
                  </li>
                </ul>
              </div>
              <div
                v-else-if="
                  audioResult.overall_classification === '转写成功，无敏感信息' // 对应转写成功但没发现敏感词的情况
                "
              >
                转录文本中未发现敏感信息。
              </div>
              <div
                v-else-if="audioResult.overall_classification === '处理失败'"
              >
                文件处理失败。
              </div>

              <el-divider v-if="index < analysisResults.length - 1" />
            </div>
          </div>
        </div>
        <div
          v-else-if="
            !loading && fileList.length > 0 && analysisResults.length === 0
          "
          class="no-results-placeholder"
        >
          <h3>检测结果</h3>
          <p>请点击“开始检测”处理文件。</p>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage, ElIcon } from 'element-plus'
import { Loading } from '@element-plus/icons-vue' // 导入加载图标

const uploadRef = ref(null) // 用于访问 el-upload 实例
// fileList 的类型是 UploadFiles (Element Plus type)，存储待上传文件
const fileList = ref([])
// analysisResults 存储后端返回的每个音频文件的分析结果，类型应为 Array<AudioProcessingResult>
// AudioProcessingResult 结构: { filename: string, transcribed_text: string | null, overall_classification: string, sensitive_segments: Array<{ text: string, category: string }> }
const analysisResults = ref([])
const loading = ref(false) // 加载状态

// **修改:** 后端语音处理接口的 URL
// 确保与后端 main.py 中注册的语音路由前缀和端点路径一致
const uploadUrl = 'http://127.0.0.1:8000/audio_classify/classify' // 后端路由前缀 + 端点 "/classify"

// **修改:** 允许上传的音频文件类型 (与后端 transcribe_audio 支持的类型一致)
const allowedFileTypes = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm']
// 单个文件最大限制（例如 50MB）
const MAX_FILE_SIZE_MB = 50
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

// 文件选择状态改变时的回调
// uploadFile 的类型是 UploadFile, uploadFiles 的类型是 UploadFiles (UploadFile 数组)
const handleFileChange = (uploadFile, uploadFiles) => {
  const file = uploadFile.raw
  if (!file) {
    return
  }

  // 校验文件类型
  const fileExtension = '.' + file.name.split('.').pop().toLowerCase()
  if (!allowedFileTypes.includes(fileExtension)) {
    ElMessage.error(
      `不支持的音频文件类型: "${fileExtension}"。请上传 ${allowedFileTypes.join(
        ', '
      )} 格式的文件。`
    )
    console.warn(`File ${file.name} has unsupported type.`)
    // Note: In on-change, the file is already added. To prevent adding, use :before-upload hook.
    return
  }

  // 校验文件大小
  if (file.size > MAX_FILE_SIZE_BYTES) {
    ElMessage.error(`文件 "${file.name}" 大小超过 ${MAX_FILE_SIZE_MB}MB 限制！`)
    console.warn(`File ${file.name} exceeds size limit.`)
    // Note: In on-change, the file is already added. To prevent adding, use :before-upload hook.
    return
  }

  // If validation passes, the file is already added to fileList.value by El-upload's v-model
}

// 文件移除时的回调
const handleFileRemove = (uploadFile, uploadFiles) => {
  // fileList.value is already updated by v-model
  console.log(`File ${uploadFile.name} removed.`)
  // If fileList becomes empty after removal, clear results
  if (fileList.value.length === 0) {
    analysisResults.value = []
  }
}

// 文件数量超出限制时的回调
const handleExceed = (files, uploadFiles) => {
  ElMessage.warning(`最多只能选择 ${uploadFiles.length} 个文件。`) // uploadFiles.length is current list size
}

// 提交上传和检测
const submitUpload = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先选择要上传的音频文件')
    return
  }

  loading.value = true
  analysisResults.value = [] // Clear previous results before starting

  const formData = new FormData()
  fileList.value.forEach((fileItem) => {
    // fileItem.raw is the native File object
    // 'files' must match the parameter name in the backend FastAPI endpoint (files: List[UploadFile] = File(...))
    formData.append('files', fileItem.raw)
  })

  try {
    const response = await axios.post(uploadUrl, formData, {
      headers: {
        'Content-Type': 'multipart/form-data' // Tell backend it's a file upload
        // 'Authorization': `Bearer ${yourAuthToken}` // Add auth header if needed
      }
      // Optional: track upload progress - uncomment if needed
      // onUploadProgress: progressEvent => {
      //    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      //    console.log(`Upload progress: ${percentCompleted}%`);
      //    // Update a progress bar state if you have one
      // }
    })

    // Check if backend response is successful (HTTP status 2xx)
    if (response.status >= 200 && response.status < 300) {
      // **修改:** 检查后端返回的数据结构是否符合预期 ({ "results": [...] })
      // 后端返回的是 AudioClassificationResponse
      if (response.data && Array.isArray(response.data.results)) {
        // **修改:** 从 response.data.results 获取音频结果列表
        analysisResults.value = response.data.results
        ElMessage.success('语音检测完成！')
      } else {
        // Backend returned success status but unexpected data structure
        ElMessage.error('后端返回结果结构不正确')
        console.error('Unexpected backend response structure:', response.data)
        // Create failure entries for each uploaded file using the NEW structure
        analysisResults.value = fileList.value.map((fileItem) => ({
          filename: fileItem.name,
          transcribed_text: null, // Set transcribed_text to null/undefined/empty string on failure
          overall_classification: '处理失败', // Frontend determined failure
          sensitive_segments: [] // No sensitive segments
        }))
      }
    } else {
      // HTTP status code is not 2xx, but within try block (less common for API errors)
      ElMessage.error(`请求返回非成功状态码: ${response.status}`)
      console.error(
        'Request returned non-success status code:',
        response.status,
        response.data
      )
      // Mark all files as failed using the NEW structure
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        transcribed_text: null,
        overall_classification: `请求失败: ${response.status}`,
        sensitive_segments: []
      }))
    }
  } catch (error) {
    console.error('语音检测请求失败:', error)
    let errorMessage = '语音检测请求失败'

    // Check for backend response (server returned non-2xx status with response body)
    if (error.response) {
      // Request was made and server responded with a status code that falls out of the range of 2xx
      // Try to get detailed error message from response body (FastAPI HTTPException usually in detail field)
      const errorDetail = error.response.data?.detail || error.message
      errorMessage += `: ${error.response.status} - ${errorDetail}`

      // **修改:** 如果后端在错误响应中包含了部分结果 (虽然我们后端目前没这样做，但作为健壮性处理)
      // Check for the NEW results structure in error response
      if (error.response.data && Array.isArray(error.response.data.results)) {
        analysisResults.value = error.response.data.results // Use the partial results if available
        ElMessage.warning(`部分文件处理可能失败: ${errorDetail}`) // Use warning because some files might have results
      } else {
        // If error response body doesn't contain results list, assume the whole batch failed
        // Mark all original uploaded files as failed using the NEW structure
        analysisResults.value = fileList.value.map((fileItem) => ({
          filename: fileItem.name,
          transcribed_text: null,
          overall_classification: '处理失败', // Frontend determined failure
          sensitive_segments: []
          // detail: errorDetail // Include backend error detail - requires template update to show
        }))
        ElMessage.error(errorMessage) // Show the main error message
      }
    } else if (error.request) {
      // Request was made but no response was received (network error, backend not running, etc.)
      errorMessage += ': 后端服务无响应或网络错误'
      // Mark all files as connection failed using the NEW structure
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        transcribed_text: null,
        overall_classification: '连接失败', // Frontend determined connection failure
        sensitive_segments: []
        // detail: '未能连接到后端服务' // requires template update
      }))
      ElMessage.error(errorMessage)
    } else {
      // Something happened in setting up the request that triggered an Error
      errorMessage += `: ${error.message}`
      // Mark all files as client error using the NEW structure
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        transcribed_text: null,
        overall_classification: '客户端错误', // Frontend determined client error
        sensitive_segments: []
        // detail: error.message // requires template update
      }))
      ElMessage.error(errorMessage)
    }
  } finally {
    loading.value = false
    // Optional: clear file list after submission regardless of success/failure
    // uploadRef.value.clearFiles();
    // fileList.value = [];
  }
}

// 清空文件列表和结果
const clearResults = () => {
  loading.value = false // Ensure loading state is off
  analysisResults.value = [] // Clear results list
  if (uploadRef.value) {
    uploadRef.value.clearFiles() // Clear El-upload internal file list
  }
  fileList.value = [] // Clear our synced state list
  ElMessage.info('已清空。')
}

// === CSS 样式保持不变或根据需要微调 ===
// 分类标签颜色样式映射（与后端四类一致）
// 这些样式可以直接复用您文档检测组件中的 <style scoped> 内容
// 只需要确保您在新的组件中也复制了这些样式定义
</script>

<style scoped>
/* 复制您文档检测组件中的 <style scoped> 内容到这里 */
/* 根据新的 HTML 结构微调选择器，例如 .document-result-item 改为 .audio-result-item */
/* .sensitive-findings 改为 .sensitive-segments 等 */
/* 添加 .transcribed-text-block 的样式 */

.app {
  max-width: 900px; /* 适当增加宽度以容纳更多内容 */
  margin: 30px auto;
  display: flex;
  justify-content: center;
}

.card {
  width: 100%;
  display: flex; /* 使用 flex 布局 */
  flex-direction: column; /* 垂直排列子元素 */
  min-height: 600px; /* 确保最小高度 */
  padding: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-bottom: 10px;
  flex-shrink: 0; /* 不压缩头部 */
}
.card-header h2 {
  margin: 0;
  font-size: 2em;
  color: #333;
}

.audio-processing-area {
  /* 修改类名 */
  flex-grow: 1; /* 占据剩余空间 */
  overflow-y: auto; /* 允许内容滚动 */
  padding: 10px 0; /* 为滚动区域顶部和底部留些空间 */
}

.upload-demo {
  margin-bottom: 15px;
}

.action-buttons {
  margin-bottom: 20px;
}

.upload-button {
  margin-right: 10px;
}

.analysis-results-area h3 {
  margin-top: 20px;
  margin-bottom: 15px;
  font-size: 1.5em;
  color: #555;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

.loading-indicator {
  text-align: center;
  font-size: 1.2em;
  color: #409eff;
  margin-top: 30px;
}
.loading-indicator .el-icon {
  vertical-align: middle;
  margin-right: 5px;
}

.no-results-placeholder {
  text-align: center;
  color: #999;
  margin-top: 50px;
}
.no-results-placeholder h3 {
  color: #999;
  margin-bottom: 10px;
}

.audio-result-item {
  /* 修改类名 */
  border: 1px solid #ddd;
  border-radius: 5px;
  padding: 15px;
  margin-bottom: 20px;
  background-color: #fff;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
}

.result-header {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #333;
  word-break: break-all; /* 长文件名换行 */
  display: flex;
  align-items: center;
  flex-wrap: wrap; /* 允许文件名和标签换行 */
}
.result-header strong {
  flex-shrink: 0; /* 文件名标签不压缩 */
  margin-right: 5px;
}

/* 新增或修改：转录文本块样式 */
.transcribed-text-block {
  margin-top: 15px;
  margin-bottom: 15px; /* Add some space after the block */
  border: 1px solid #eee;
  border-left: 4px solid #409eff; /* Add a colored border to the side */
  padding: 10px;
  background-color: #f9f9f9; /* Slightly different background */
  border-radius: 4px;
}
.transcribed-text-block h4 {
  margin-top: 0;
  margin-bottom: 5px;
  font-size: 1.1em;
  color: #555;
}
.transcribed-text-block pre {
  white-space: pre-wrap; /* Preserve line breaks and wrap text */
  word-break: break-all; /* Break long words */
  font-size: 0.9em;
  color: #333;
  font-family: Consolas, Menlo, Monaco, 'Courier New', monospace; /* Monospace font */
  max-height: 200px; /* Limit height and add scroll if needed */
  overflow-y: auto;
  padding: 0; /* Remove pre default padding */
  margin: 0; /* Remove pre default margin */
}

/* 修改：敏感发现列表的类名 */
.segments-list {
  list-style: none;
  padding: 0;
  margin: 0;
  margin-top: 10px; /* Add space above findings list */
}

/* 修改：敏感发现项的类名 */
.segment-item {
  border: 1px dashed #ccc;
  border-radius: 4px;
  padding: 10px;
  margin-bottom: 10px;
  background-color: #fefefe;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.segment-item:last-child {
  margin-bottom: 0;
}

/* 修改：敏感片段类别和文本片段的类名 */
.segment-category,
.segment-snippet {
  font-size: 0.95em;
  margin-bottom: 5px;
}
.segment-category strong,
.segment-snippet strong {
  color: #666;
  margin-right: 5px;
}

.segment-snippet pre {
  /* 修改类名 */
  background: #fff;
  padding: 8px;
  border: 1px solid #eee;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 0.9em;
  max-height: 100px;
  overflow-y: auto;
  font-family: Consolas, Menlo, Monaco, 'Courier New', monospace;
  color: #333;
  margin: 0; /* Remove default pre margin */
}

/* === 分类标签颜色样式（与后端四类一致）=== */
/* 这部分样式可以直接复用，因为后端返回的敏感类别名称是一样的 */
.category-label {
  font-weight: bold;
  padding: 3px 8px;
  border-radius: 4px;
  color: #fff;
  background: #999; /* Default grey */
  font-size: 0.9em;
  display: inline-block;
  margin-top: 0;
  white-space: nowrap;
  vertical-align: middle;
  margin-left: 10px;
}
/* 总体分类标签的颜色使用 category-label 类的背景颜色 */
.category-personal-privacy {
  background: #e6a23c; /* Orange/Yellowish for privacy */
}
.category-porn {
  background: #f56c6c; /* Red for porn */
}
.category-bad-harmful {
  background: #409eff; /* Blue for combined bad/harmful */
}
.category-non-sensitive {
  background: #67c23a; /* Green for non-sensitive */
}
.category-failed {
  /* 用于后端返回 overall_classification 为非以上四类的情况 */
  background: #a1a1a1; /* Darker grey */
}
/* === CSS 结束 === */

/* Element Plus 覆盖样式 */
.el-upload__tip {
  font-size: 0.9em;
  color: #999;
  margin-top: 7px;
}
</style>
