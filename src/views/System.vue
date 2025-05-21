<template>
  <div class="app">
    <el-card class="card">
      <div class="card-header">
        <h2>文档敏感信息检测</h2>
      </div>

      <div class="document-processing-area">
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
          <el-button type="primary">选择文件</el-button>
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
              v-for="(docResult, docIndex) in analysisResults"
              :key="docResult.filename + '_' + docIndex"
              class="document-result-item"
            >
              <div class="result-header">
                <strong>文件名:</strong> {{ docResult.filename }}
                <span
                  :class="{
                    'category-label': true,
                    'category-personal-privacy':
                      docResult.overall_classification === '个人隐私信息',
                    'category-porn':
                      docResult.overall_classification === '黄色信息',
                    'category-bad-harmful':
                      docResult.overall_classification === '不良有害信息',
                    'category-non-sensitive':
                      docResult.overall_classification === '无敏感信息',
                    'category-failed':
                      docResult.overall_classification === '分类失败' // 后端分类失败时的显示
                    // 如果有前端处理失败的情况，也可以添加如 'status-failed' 类
                    // 'status-failed': docResult.status === 'failed' // 例如：用于标记网络错误等前端检测到的失败
                  }"
                >
                  {{ docResult.overall_classification || '处理失败' }}
                </span>
              </div>

              <div
                v-if="
                  docResult.overall_classification !== '无敏感信息' &&
                  Array.isArray(docResult.sensitive_findings) &&
                  docResult.sensitive_findings.length > 0
                "
              >
                <h4>
                  敏感信息发现 ({{ docResult.sensitive_findings.length }} 条):
                </h4>
                <ul class="findings-list">
                  <li
                    v-for="(finding, findIndex) in docResult.sensitive_findings"
                    :key="findIndex"
                    class="finding-item"
                  >
                    <div class="finding-category">
                      <strong>类别:</strong>
                      <span
                        :class="{
                          'category-label': true,
                          'category-personal-privacy':
                            finding.category === '个人隐私信息',
                          'category-porn': finding.category === '黄色信息',
                          'category-bad-harmful':
                            finding.category === '不良有害信息',
                          // 'category-non-sensitive': finding.category === '无敏感信息', // 敏感信息发现列表里不展示“无敏感信息”的片段
                          'category-failed': finding.category === '分类失败' // 虽然不应该出现，作为后备
                        }"
                      >
                        {{ finding.category }}
                      </span>
                    </div>
                    <div class="finding-snippet">
                      <strong>片段:</strong>
                      <pre>{{ finding.text }}</pre>
                    </div>
                  </li>
                </ul>
              </div>
              <div
                v-else-if="docResult.overall_classification === '无敏感信息'"
              >
                未发现敏感信息。
              </div>
              <div v-else>处理结果未知或分类失败。</div>

              <el-divider v-if="docIndex < analysisResults.length - 1" />
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
// analysisResults 存储后端返回的每个文档的分析结果，类型应为 Array<DocumentClassificationResult>
// DocumentClassificationResult 结构: { filename: string, overall_classification: string, sensitive_findings: Array<{ text: string, category: string }> }
const analysisResults = ref([])
const loading = ref(false) // 加载状态

// 修改：后端文档处理接口的 URL
// 确保与后端 main.py 中注册的路由前缀和端点路径一致
// 修改：后端文档处理接口的 URL
// 确保与后端 main.py 中注册的路由前缀和 classification.py 中的端点路径一致
const uploadUrl = 'http://127.0.0.1:8000/document_classify/classify' // 后端路由前缀 + 端点 "/classify" // 后端路由前缀 + 端点 "/"

// 允许上传的文件类型 (与后端 extract_text_from_document 支持的类型一致)
const allowedFileTypes = ['.txt', '.docx', '.pdf']
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
      `不支持的文件类型: "${fileExtension}"。请上传 ${allowedFileTypes.join(
        ', '
      )} 格式的文件。`
    )
    console.warn(`File ${file.name} has unsupported type.`)
    // 如果需要严格阻止添加，可以使用 :before-upload hook 并返回 false
    // return false; // 如果在 before-upload 中，返回 false 会阻止文件加入列表
    // 在 on-change 中，文件已被加入，只能提示错误
    return
  }

  // 校验文件大小
  if (file.size > MAX_FILE_SIZE_BYTES) {
    ElMessage.error(`文件 "${file.name}" 大小超过 ${MAX_FILE_SIZE_MB}MB 限制！`)
    console.warn(`File ${file.name} exceeds size limit.`)
    // 如果需要严格阻止添加，可以使用 :before-upload hook 并返回 false
    // return false; // 如果在 before-upload 中，返回 false 会阻止文件加入列表
    // 在 on-change 中，文件已被加入，只能提示错误
    return
  }

  // 如果校验通过，文件已被 El-upload 加入 fileList.value
}

// 文件移除时的回调
const handleFileRemove = (uploadFile, uploadFiles) => {
  // fileList.value 已经由 v-model 更新
  console.log(`File ${uploadFile.name} removed.`)
  // 如果移除文件后 fileList 为空，可以考虑清空 analysisResults
  if (fileList.value.length === 0) {
    analysisResults.value = []
  }
}

// 文件数量超出限制时的回调
const handleExceed = (files, uploadFiles) => {
  ElMessage.warning(`最多只能选择 ${uploadFiles.length} 个文件。`) // uploadFiles.length 是当前列表中的文件数
}

// 提交上传和检测
const submitUpload = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先选择要上传的文件')
    return
  }

  loading.value = true
  analysisResults.value = [] // 在开始处理前清空之前的检测结果

  const formData = new FormData()
  fileList.value.forEach((fileItem) => {
    // fileItem.raw 是原生的 File 对象
    formData.append('files', fileItem.raw) // 'files' 必须与后端 FastAPI 接口参数名一致 (files: List[UploadFile] = File(...))
  })

  try {
    const response = await axios.post(uploadUrl, formData, {
      headers: {
        'Content-Type': 'multipart/form-data' // 告诉后端是文件上传
        // 如果后端需要认证，可能还需要添加认证 header
        // 'Authorization': `Bearer ${yourAuthToken}`
      }
      // Optional: track upload progress - uncomment if needed
      // onUploadProgress: progressEvent => {
      //    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      //    console.log(`Upload progress: ${percentCompleted}%`);
      //    // Update a progress bar state if you have one
      // }
    })

    // 检查后端响应是否成功 (HTTP status 2xx)
    if (response.status >= 200 && response.status < 300) {
      // 修改：检查后端返回的数据结构是否符合预期 ({ "documents": [...] })
      // 后端返回的是 MultiDocumentClassificationResult
      if (response.data && Array.isArray(response.data.documents)) {
        // 修改：从 response.data.documents 获取文档结果列表
        analysisResults.value = response.data.documents
        ElMessage.success('文档检测完成！')
      } else {
        // 后端返回结构不符合预期，但 HTTP 状态码是成功的
        ElMessage.error('后端返回结果结构不正确')
        console.error('Unexpected backend response structure:', response.data)
        // 为每个上传的文件创建一个失败状态条目，以便用户知道哪些文件处理失败了
        analysisResults.value = fileList.value.map((fileItem) => ({
          filename: fileItem.name,
          overall_classification: '处理失败', // 用 Overall classification 来表示前端识别到的处理失败
          sensitive_findings: [] // 没有敏感发现
          // detail: '后端返回数据结构异常' // 可以添加 detail 字段用于显示错误详情，但需要调整模板显示
        }))
      }
    } else {
      // HTTP 状态码不是 2xx，但在 Axios 的 try 块中（通常是 3xx 或重定向，不常见于 API 错误）
      // 这种情况理论上较少发生，除非后端配置有问题
      ElMessage.error(`请求返回非成功状态码: ${response.status}`)
      console.error(
        'Request returned non-success status code:',
        response.status,
        response.data
      )
      // 为每个上传的文件标记失败
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        overall_classification: `请求失败: ${response.status}`,
        sensitive_findings: []
        // detail: `HTTP 状态码 ${response.status}`
      }))
    }
  } catch (error) {
    console.error('文档检测请求失败:', error)
    let errorMessage = '文档检测请求失败'

    // 检查是否有后端响应 (服务器返回了非 2xx 状态码，且有响应体)
    if (error.response) {
      // 请求已发送，服务器响应了非 2xx 状态码
      // 尝试从响应体中获取详细错误信息 (FastAPI HTTPException 通常在 detail 字段)
      const errorDetail = error.response.data?.detail || error.message
      errorMessage += `: ${error.response.status} - ${errorDetail}`

      // 如果后端在错误响应中包含了部分文档的处理结果 (虽然我们后端目前没这样做，但可以作为一种健壮性处理)
      if (error.response.data && Array.isArray(error.response.data.documents)) {
        analysisResults.value = error.response.data.documents
        ElMessage.warning(`部分文件处理可能失败: ${errorDetail}`) // 使用 warning 因为有些文件可能有结果
      } else {
        // 如果错误响应体中没有文档结果列表，则认为整个批次处理失败
        // 为所有原上传文件标记为处理失败
        analysisResults.value = fileList.value.map((fileItem) => ({
          filename: fileItem.name,
          overall_classification: '处理失败', // Frontend determined failure
          sensitive_findings: []
          // detail: errorDetail // Include backend error detail
        }))
        ElMessage.error(errorMessage) // 显示主要错误信息
      }
    } else if (error.request) {
      // 请求已发送，但没有收到响应 (网络错误，后端服务未运行等)
      errorMessage += ': 后端服务无响应或网络错误'
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        overall_classification: '连接失败', // Frontend determined connection failure
        sensitive_findings: []
        // detail: '未能连接到后端服务'
      }))
      ElMessage.error(errorMessage)
    } else {
      // 其他类型的错误 (例如，设置请求时出错)
      errorMessage += `: ${error.message}`
      analysisResults.value = fileList.value.map((fileItem) => ({
        filename: fileItem.name,
        overall_classification: '客户端错误', // Frontend determined client error
        sensitive_findings: []
        // detail: error.message
      }))
      ElMessage.error(errorMessage)
    }
  } finally {
    loading.value = false
    // 清空文件列表的时机：通常在成功上传并处理完成后清空
    // 如果希望不论成功失败都清空，可以在这里取消注释
    // uploadRef.value.clearFiles();
    // fileList.value = []; // 同步更新状态
  }
}

// 清空文件列表和结果
const clearResults = () => {
  loading.value = false // 确保加载状态关闭
  analysisResults.value = [] // 清空结果列表
  if (uploadRef.value) {
    uploadRef.value.clearFiles() // 清空 El-upload 内部的文件列表
  }
  fileList.value = [] // 清空我们同步的状态列表
  ElMessage.info('已清空。')
}

// === CSS 样式保持不变 ===
// 分类标签颜色样式映射（与后端四类一致）
// CSS 定义在 <style scoped> 中，与您提供的一致，无需修改
</script>

<style scoped>
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

.document-processing-area {
  flex-grow: 1; /* 占据剩余空间 */
  overflow-y: auto; /* 允许内容滚动 */
  padding: 10px 0; /* 为滚动区域顶部和底部留些空间 */
}

.upload-demo {
  margin-bottom: 15px; /* Adjusted margin */
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

.document-result-item {
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

/* 修改：不再使用 status-label 类，直接使用 category-label 显示总体分类 */
/* .status-label {
  display: inline-block;
  margin-left: 10px;
  padding: 3px 8px;
  border-radius: 4px;
  color: #fff;
  font-size: 0.9em;
  font-weight: bold;
}
.status-processing {
  background-color: #409eff;
}
.status-success {
  background-color: #67c23a;
}
.status-failed {
  background-color: #f56c6c;
}
.status-unsupported_format {
  background-color: #e6a23c;
}
.status-unknown {
  background-color: #909399;
} */

/* result-detail 不再由后端返回，移除相关显示逻辑 */
/* .result-detail {
  font-size: 0.95em;
  color: #666;
  margin-bottom: 10px;
} */

/* extracted-text-preview, total-text-info 不再由后端返回，移除相关显示逻辑 */
/* .extracted-text-preview,
.total-text-info {
  margin-top: 10px;
  font-size: 0.9em;
  color: #555;
}
.extracted-text-preview pre {
  background: #f0f0f0;
  padding: 10px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 150px;
  overflow-y: auto;
  margin-top: 5px;
  font-family: Consolas, Menlo, Monaco, 'Courier New', monospace;
  color: #333;
} */

.sensitive-findings h4 {
  margin-top: 15px;
  margin-bottom: 10px;
  font-size: 1.1em;
  color: #444;
}

.findings-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.finding-item {
  border: 1px dashed #ccc;
  border-radius: 4px;
  padding: 10px;
  margin-bottom: 10px;
  background-color: #fefefe;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); /* Subtle shadow */
}
.finding-item:last-child {
  margin-bottom: 0;
}

/* finding-location 不再由后端返回，移除相关显示逻辑 */
/* .finding-location, */
.finding-category {
  font-size: 0.95em;
  margin-bottom: 5px;
}
/* finding-location strong, */
.finding-category strong,
.finding-snippet strong {
  color: #666;
  margin-right: 5px;
}

.finding-snippet pre {
  background: #fff;
  padding: 8px;
  border: 1px solid #eee;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 0.9em;
  max-height: 100px; /* 限制片段预览高度 */
  overflow-y: auto; /* 片段文本过多时可滚动 */
  font-family: Consolas, Menlo, Monaco, 'Courier New', monospace;
  color: #333; /* Darker text */
}

/* === 分类标签颜色样式（与后端四类一致）=== */
.category-label {
  font-weight: bold;
  padding: 3px 8px;
  border-radius: 4px;
  color: #fff;
  background: #999; /* Default grey */
  font-size: 0.9em;
  display: inline-block;
  margin-top: 0; /* Adjusted margin */
  white-space: nowrap;
  vertical-align: middle; /* Vertical align with text */
  margin-left: 10px; /* Add margin to the left of the label */
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
  /* 用于后端返回 overall_classification 为 '分类失败' 或前端标记为失败的情况 */
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
