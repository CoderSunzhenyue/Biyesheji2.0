<template>
  <div class="upload-container">
    <el-card class="card">
      <template #header>
        <span>上传图片进行敏感信息检测</span>
      </template>

      <el-upload
        class="upload-area"
        drag
        multiple
        :action="null"
        :auto-upload="false"
        :on-change="handleFileChange"
        :file-list="fileList"
        accept="image/*"
        list-type="picture-card"
        :show-file-list="false"
      >
        <el-icon><Upload /></el-icon>
        <div class="el-upload__text">拖拽图片或点击上传</div>
        <template #tip>
          <div class="el-upload__tip">
            支持上传 jpg/png/bmp 等格式图片，单次可选择多张。
          </div>
        </template>
      </el-upload>
    </el-card>

    <el-card v-if="fileList.length > 0" style="margin-top: 20px">
      <template #header>
        <span>已选择图片列表 (共 {{ fileList.length }} 张)</span>
      </template>

      <el-table :data="fileList" style="width: 100%">
        <el-table-column type="index" label="#" width="50"></el-table-column>

        <el-table-column label="文件">
          <template #default="scope">
            <div style="display: flex; align-items: center">
              <span>{{ scope.row.name }}</span>
              <el-tag
                v-if="scope.row.detectionStatus === 'detecting'"
                type="info"
                size="small"
                style="margin-left: 10px"
              >
                检测中...
              </el-tag>
              <el-tag
                v-if="scope.row.detectionStatus === 'detected'"
                type="success"
                size="small"
                style="margin-left: 10px"
              >
                已检测
              </el-tag>
              <el-tag
                v-if="scope.row.detectionStatus === 'error'"
                type="danger"
                size="small"
                style="margin-left: 10px"
              >
                检测失败
              </el-tag>
              <el-tag
                v-if="scope.row.detectionStatus === 'pending'"
                type="default"
                size="small"
                style="margin-left: 10px"
              >
                待检测
              </el-tag>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="280">
          <template #default="scope">
            <el-button
              size="small"
              type="primary"
              :disabled="scope.row.detectionStatus === 'detecting'"
              @click="detectSingleFile(scope.row)"
            >
              {{
                scope.row.detectionStatus === 'detecting'
                  ? '检测中'
                  : '开始检测'
              }}
            </el-button>
            <el-button
              size="small"
              type="info"
              :disabled="scope.row.detectionStatus !== 'detected'"
              @click="viewResult(scope.row)"
            >
              查看结果
            </el-button>
            <el-button
              size="small"
              type="danger"
              @click="deleteFile(scope.row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="showResultModal"
      :title="
        currentResultFile ? `检测结果 - ${currentResultFile.name}` : '检测结果'
      "
      width="800px"
      top="50px"
      destroy-on-close
    >
      <div v-if="currentResultFile && currentResultFile.detectionResult">
        <div class="color-legend">
          <strong>边框颜色说明:</strong>
          <ul>
            <li><span class="color-swatch red"></span> 个人隐私信息</li>
            <li><span class="color-swatch yellow"></span> 黄色信息</li>
            <li><span class="color-swatch blue"></span> 不良有害信息</li>
            <li><span class="color-swatch green"></span> 无敏感信息</li>
          </ul>
        </div>
        <el-image
          :src="currentResultFile.annotatedImageUrl"
          style="
            width: 100%;
            max-height: 500px;
            margin-bottom: 20px;
            border: 1px solid #eee;
          "
          fit="contain"
          loading="lazy"
          alt="Annotated Image"
        ></el-image>

        <div
          v-if="
            currentResultFile.detectionResult.detections &&
            currentResultFile.detectionResult.detections.length > 0
          "
        >
          <h4>检测到的文本列表:</h4>
          <el-table
            :data="currentResultFile.detectionResult.detections"
            style="width: 100%"
            max-height="300"
            stripe
          >
            <el-table-column
              type="index"
              label="#"
              width="50"
            ></el-table-column>
            <el-table-column
              prop="text"
              label="文本"
              show-overflow-tooltip
            ></el-table-column>
            <el-table-column prop="category" label="分类" width="120">
              <template #default="scope">
                <el-tag
                  :type="
                    scope.row.category === '个人隐私信息'
                      ? 'danger'
                      : scope.row.category === '黄色信息'
                      ? 'warning'
                      : scope.row.category === '不良有害信息'
                      ? 'primary'
                      : scope.row.category === '无敏感信息'
                      ? 'success'
                      : 'info'
                  "
                  size="small"
                >
                  {{ scope.row.category }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </div>
        <div v-else>
          <p>该图片未检测到敏感信息或文本。</p>
        </div>
      </div>
      <div v-else>
        <p>加载结果失败或没有可显示的结果。</p>
      </div>
      <template #footer>
        <el-button @click="showResultModal = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage, ElSkeleton, ElDialog, ElTag } from 'element-plus' // 导入需要的 Element Plus 组件
import { Upload } from '@element-plus/icons-vue'

// 响应式状态变量
const fileList = ref([]) // 上传的文件列表，每个文件对象会添加检测状态和结果
const showResultModal = ref(false) // 控制检测结果弹窗的显示隐藏
const currentResultFile = ref(null) // 当前在弹窗中显示结果的文件对象

// 处理文件选择和文件列表变化
const handleFileChange = (file, files) => {
  // 过滤非图片文件
  if (!file.raw || !file.raw.type.startsWith('image/')) {
    ElMessage.warning(`文件 ${file.name} 不是图片格式，请重新选择。`)
    // 移除不符合的文件
    const index = files.indexOf(file)
    if (index > -1) {
      files.splice(index, 1)
    }
  }

  // 为每个新添加的有效文件初始化检测状态和结果字段
  // 遍历当前文件列表，为新文件添加属性
  const currentValidFiles = files.filter(
    (f) => f.raw && f.raw.type.startsWith('image/')
  )
  currentValidFiles.forEach((file) => {
    // 如果是新文件（或者还没有初始化过），则添加属性
    if (file.detectionStatus === undefined) {
      // 使用 undefined 来判断是否是新文件或未初始化
      file.detectionStatus = 'pending' // 初始状态：待检测
      file.detectionResult = null // 初始化检测结果
      file.annotatedImageUrl = null // 初始化标注图片URL
      // file.url = URL.createObjectURL(file.raw); // 如果需要本地预览小图，可以生成本地URL
    }
  })

  // 更新文件列表 ref，触发视图更新
  fileList.value = [...currentValidFiles] // 使用展开运算符确保响应式更新
}

// 从列表中删除文件
const deleteFile = (file) => {
  const index = fileList.value.findIndex((f) => f.uid === file.uid)
  if (index > -1) {
    // if (file.url) { URL.revokeObjectURL(file.url); } // 如果生成了本地URL，记得释放
    fileList.value.splice(index, 1) // 从文件列表中移除
    ElMessage.success(`${file.name} 已移除.`)
  }
}

// 对单个文件进行敏感信息检测
const detectSingleFile = async (file) => {
  // 防止重复检测
  if (file.detectionStatus === 'detecting') {
    console.log(`${file.name} 正在检测中，请勿重复点击.`)
    return
  }

  // 更新文件状态为检测中
  file.detectionStatus = 'detecting'
  file.detectionResult = null // 清空旧结果
  file.annotatedImageUrl = null // 清空旧图片URL

  const formData = new FormData()
  formData.append('files', file.raw) // 将原始文件添加到 FormData

  // 显示针对该文件的加载提示
  const fileLoadingMsg = ElMessage({
    message: `正在检测图片 ${file.name}，请稍候...`,
    type: 'info',
    duration: 0, // 不会自动关闭
    key: `detecting_${file.uid}` // 使用文件uid作为key，方便管理
  })

  try {
    // 调用后端API，这里假设后端接收 multipart/form-data 并且期望一个或多个文件在 'files' 字段
    // 并返回一个包含 'images' 数组的JSON，数组中每个元素对应一个文件的检测结果，
    // 结果对象包含 filename, detections (文本列表), annotated_image (标注图片Base64 URL)
    const res = await axios.post('http://localhost:8000/api/detect', formData, {
      headers: {
        // 假设你的后端需要JWT认证，根据实际情况调整或移除
        Authorization: `Bearer ${localStorage.getItem('access_token')}`,
        'Content-Type': 'multipart/form-data' // 必须设置
      },
      timeout: 60000 // 设置请求超时时间，例如 60 秒
    })

    // 检查后端返回的数据结构是否符合预期
    if (
      res.data &&
      Array.isArray(res.data.images) &&
      res.data.images.length > 0
    ) {
      // 假设后端返回的第一个结果就是当前文件的结果
      const result = res.data.images[0]
      file.detectionResult = result // 存储检测结果对象
      file.annotatedImageUrl = result.annotated_image // 存储标注图片Base64 URL

      file.detectionStatus = 'detected' // 更新状态为已检测
      ElMessage.success(`${file.name} 检测完成！`)
    } else {
      console.error(
        'Backend response data structure for single file is unexpected:',
        res.data
      )
      file.detectionStatus = 'error' // 更新状态为检测失败
      ElMessage.error(`${file.name} 检测失败：后端返回数据格式错误或为空。`)
    }
  } catch (error) {
    console.error(`Detection error for ${file.name}:`, error)
    // 构造更友好的错误信息
    const errorMessage =
      error.response?.data?.detail || // 尝试获取后端返回的错误详情
      error.message || // axios或其他网络错误信息
      `检测图片 ${file.name} 失败，请检查网络或后端服务是否正常运行。` // 通用错误
    file.detectionStatus = 'error' // 更新状态为检测失败
    ElMessage.error(errorMessage)
  } finally {
    // 确保针对该文件的加载提示被关闭
    if (fileLoadingMsg) {
      fileLoadingMsg.close()
    }
    // 如果有全局加载状态，也在这里处理其逻辑
  }
}

// 查看检测结果，打开弹窗
const viewResult = (file) => {
  // 只有当文件状态为 'detected' 且有结果数据时才显示弹窗
  if (file.detectionStatus === 'detected' && file.detectionResult) {
    currentResultFile.value = file // 将当前文件对象赋值给状态变量
    showResultModal.value = true // 显示弹窗
  } else {
    ElMessage.warning(`请先对图片 ${file.name} 进行检测。`)
  }
}

// 注意：原有的全局 submitDetection 按钮被移除。
// 如果需要一个按钮来一次性检测所有“待检测”状态的文件，
// 可以重新添加该按钮，并修改其逻辑为遍历 fileList.value 中
// detectionStatus === 'pending' 的文件，并依次调用 detectSingleFile(file)。
</script>

<style scoped>
/* ... 其他现有的样式 ... */

.color-legend {
  margin-bottom: 20px; /* 弹窗内图例底部间距 */
  border: 1px solid #eee; /* 可选：给图例加个边框 */
  padding: 15px; /* 可选：增加内边距 */
  background-color: #f9f9f9; /* 可选：给图例加个背景色 */
  border-radius: 4px; /* 可选：圆角 */
}

.color-legend ul {
  list-style: none; /* 移除默认列表点 */
  padding: 0;
  margin: 0;
  display: flex; /* 使用 Flexbox 排列项目 */
  flex-wrap: wrap; /* 如果空间不足，允许换行 */
  gap: 15px; /* 项目之间的间距 */
}

.color-legend li {
  display: flex; /* 使 li 内容对齐 */
  align-items: center; /* 垂直居中对齐 */
  font-size: 14px;
}

.color-swatch {
  display: inline-block; /* 使 span 能够设置宽度和高度 */
  width: 18px; /* 方块的宽度 */
  height: 18px; /* 方块的高度 */
  margin-right: 8px; /* 方块和文本之间的间距 */
  border-radius: 3px; /* 可选：使方块有轻微圆角 */
  vertical-align: middle; /* 垂直对齐方式 */
  border: 1px solid #ccc; /* 可选：给方块加个浅色边框 */
}

/* 定义不同颜色的背景 */
.color-swatch.red {
  background-color: #f56c6c; /* Element Plus danger color */
}

.color-swatch.yellow {
  background-color: #e6a23c; /* Element Plus warning color */
}

.color-swatch.blue {
  background-color: #409eff; /* Element Plus primary color */
}

.color-swatch.green {
  background-color: #67c23a; /* Element Plus success color */
}

.color-swatch.gray {
  background-color: #909399; /* Element Plus info/default color */
}

/* 确保 Element Plus 组件样式也包含 */
/* .el-dialog .el-image { ... } */
/* .el-dialog .el-table { ... } */
/* ... 其他现有的 Element Plus 组件样式 ... */
</style>
