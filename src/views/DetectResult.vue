<template>
  <div class="result-container">
    <el-select
      v-model="selectedFilename"
      placeholder="请选择图片记录"
      class="select-image"
      size="large"
      :disabled="isLoadingRecords || groupedResults.length === 0"
    >
      <el-option
        v-for="(item, idx) in groupedResults"
        :key="item.filename"
        :label="`图片 ${idx + 1} (${item.filename})`"
        :value="item.filename"
      />
    </el-select>

    <div v-if="isLoadingRecords">
      <el-card class="result-card">
        <el-skeleton :rows="8" animated />
      </el-card>
    </div>
    <div v-else-if="groupedResults.length === 0">
      <el-empty description="暂无检测记录"></el-empty>
    </div>

    <el-card
      v-else-if="selectedResult"
      :key="selectedResult.filename"
      class="result-card"
    >
      <template #header>
        <div class="card-header">
          <span>
            图片 {{ currentIndex + 1 }}：{{ selectedResult.filename }}
          </span>
          <el-button
            type="danger"
            size="default"
            @click="deleteCurrentImageRecords"
            :disabled="isLoadingRecords"
          >
            删除记录
          </el-button>
        </div>
      </template>

      <el-row :gutter="20">
        <el-col :md="12" :sm="24">
          <h4 class="section-title">图片预览</h4>
          <el-image
            :src="selectedResult.saved_path"
            :alt="selectedResult.filename"
            class="preview-img"
            fit="contain"
            loading="lazy"
          >
            <template #error>
              <div class="image-slot">
                <el-icon><Picture /></el-icon>
                <span>图片加载失败</span>
              </div>
            </template>
            <template #placeholder>
              <div class="image-slot is-loading">
                <el-icon><Loading /></el-icon>
                <span>加载中...</span>
              </div>
            </template>
          </el-image>
          <div class="color-legend">
            <strong>边框颜色说明:</strong>
            <ul>
              <li><span class="color-swatch red"></span> 个人隐私信息</li>
              <li><span class="color-swatch yellow"></span> 黄色信息</li>
              <li><span class="color-swatch blue"></span> 不良有害信息</li>
              <li><span class="color-swatch green"></span> 无敏感信息</li>
            </ul>
          </div>
        </el-col>
        <el-col :md="12" :sm="24">
          <h4 class="section-title">检测到的敏感文本</h4>
          <div class="detections-list">
            <div
              v-for="(det, i) in selectedResult.detections"
              :key="i"
              class="detection-item"
              :class="`category-${getCategoryColorClass(det.category)}`"
            >
              <div class="text-result">
                <strong>识别文字：</strong>{{ det.text || '未识别到文字' }}
              </div>
              <div class="sensitive-result">
                <strong>敏感信息：</strong>
                <el-tag size="small" :type="getCategoryTagType(det.category)">
                  {{ det.category || '无' }}
                </el-tag>
              </div>
              <div class="confidence" v-if="det.confidence !== undefined">
                <strong>置信度：</strong
                ><span class="confidence-score"
                  >{{ (det.confidence * 100).toFixed(2) }}%</span
                >
              </div>
            </div>
            <div
              v-if="
                !selectedResult.detections ||
                selectedResult.detections.length === 0
              "
            >
              <el-empty
                description="该图片未检测到敏感文本"
                :image-size="50"
              ></el-empty>
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import axios from 'axios'
import {
  ElMessage,
  ElMessageBox,
  ElCard,
  ElSelect,
  ElOption,
  ElButton,
  ElEmpty,
  ElRow,
  ElCol,
  ElTag,
  ElSkeleton,
  ElImage,
  ElIcon
} from 'element-plus' // Import all used components
import { Picture, Loading } from '@element-plus/icons-vue' // Import icons if used in image slot

// --- State ---
const results = ref([]) // 后端返回的原始记录列表
const selectedFilename = ref('') // 选中的文件名
const isLoadingRecords = ref(false) // 加载记录的状态
// -------------

// --- Computed Properties ---

// 按 filename 分组，并根据后端返回的数据构造每张图片完整的记录对象
const groupedResults = computed(() => {
  const map = {}
  // 假设 'results.value' 是一个数组，每个元素可能是单条检测记录或包含多条 detections 的记录
  results.value.forEach((record) => {
    // 确保 record 是有效的，并且有 filename 属性
    if (record && record.filename) {
      // 如果 map 中还没有这个 filename，初始化它
      if (!map[record.filename]) {
        map[record.filename] = {
          filename: record.filename,
          detections: [], // 初始化 detections 数组
          // 使用后端提供的 image_url，并拼接完整的基地址
          saved_path: record.saved_path
            ? `http://127.0.0.1:8000${record.saved_path}`
            : ''
        }
      }

      // 将当前 record 的 detections 添加到对应 filename 的 detections 数组中
      // 考虑到后端返回的 record 可能本身就有一个 detections 数组，或者是一个单独的 detection 对象
      // 这里尝试将 detections 添加进去。如果后端返回结构复杂，需要根据实际情况调整。
      if (record.detections && Array.isArray(record.detections)) {
        map[record.filename].detections.push(...record.detections)
      }
      // 如果后端返回的是单条 detection 记录，并且没有 detections 数组，可以 uncomment 下面的代码
      // else if (record.text && record.category && record.bbox) {
      //    map[record.filename].detections.push({
      //       text: record.text, category: record.category, bbox: record.bbox, confidence: record.confidence // 确保 confidence 存在
      //    });
      // }

      if (record.saved_path) {
        map[
          record.filename
        ].saved_path = `http://127.0.0.1:8000${record.saved_path}` // 拼接完整地址
      }
    }
  })
  // 将 map 的值（即每个文件的完整记录对象）转换为数组并按文件名排序
  return Object.values(map).sort((a, b) => a.filename.localeCompare(b.filename))
})

// 如果 groupedResults 变化且还没选中，就默认选第一项
watch(
  groupedResults,
  (newVal) => {
    // 只有当 newVal 有数据且当前没有选中项或选中项不在新数据中时才默认选中第一个
    const isSelectedFilenameStillValid = newVal.some(
      (item) => item.filename === selectedFilename.value
    )
    if (newVal.length > 0 && !isSelectedFilenameStillValid) {
      selectedFilename.value = newVal[0].filename
    } else if (newVal.length === 0) {
      // 如果 groupedResults 为空，清空选中状态
      selectedFilename.value = ''
    }
  },
  { immediate: true } // 立即执行一次 watch，无需深度监听整个数组内容变化
)

// 根据 selectedFilename 找到对应的整条数据
const selectedResult = computed(
  () =>
    groupedResults.value.find(
      (item) => item.filename === selectedFilename.value
    ) || null // 找不到则返回null
)

// 用于显示索引（第几张图）
const currentIndex = computed(() =>
  groupedResults.value.findIndex(
    (item) => item.filename === selectedFilename.value
  )
)
// -----------------------------

// --- Methods ---

// 加载记录的函数
const fetchRecords = async () => {
  isLoadingRecords.value = true // 开始加载
  results.value = [] // 清空旧数据
  selectedFilename.value = '' // 清空选中状态
  try {
    const token = localStorage.getItem('access_token')
    if (!token) {
      ElMessage.warning('请先登录')
      return
    }
    // 确保这里的 API 路径和方法正确
    const { data } = await axios.get('http://127.0.0.1:8000/api/records/', {
      headers: { Authorization: `Bearer ${token}` }
    })

    // *** 重要 ***: 检查后端返回的数据结构。这里假设 data 是一个数组，
    // 每个元素是 {filename: '...', image_url: '...', detections: [...]}.
    // 如果后端返回结构不同 (例如 {records: [...]}), 需要调整 data 的取值。
    if (Array.isArray(data)) {
      results.value = data // 更新原始记录数据
      console.log('Fetched records raw data:', data) // 添加日志查看返回的原始数据结构
      // groupedResults computed property 会自动处理分组和格式化
      if (groupedResults.value.length === 0 && results.value.length > 0) {
        console.warn(
          'Raw data fetched, but groupedResults is empty. Check grouping logic and raw data format.'
        )
        // 如果原始数据不为空但分组后为空，可能是分组逻辑有问题或数据格式不符合预期
        ElMessage.warning('记录数据格式可能不正确，无法正常显示。')
      } else if (groupedResults.value.length > 0) {
        console.log('Grouped results:', groupedResults.value)
      }
    } else {
      console.error(
        'Backend records API response structure is unexpected:',
        data
      )
      ElMessage.error('后端记录数据格式错误。')
    }
  } catch (error) {
    console.error('Error fetching records:', error)
    ElMessage.error(
      '获取检测记录失败: ' + (error.response?.data?.detail || error.message)
    )
  } finally {
    isLoadingRecords.value = false // 结束加载
  }
}

// 删除当前图片所有记录的方法
const deleteCurrentImageRecords = async () => {
  if (!selectedFilename.value) {
    ElMessage.warning('请先选择要删除记录的图片')
    return
  }

  ElMessageBox.confirm(
    `确定要删除图片 "${selectedFilename.value}" 的所有记录吗？`,
    '警告',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  )
    .then(async () => {
      try {
        const token = localStorage.getItem('access_token')
        if (!token) {
          ElMessage.warning('请先登录')
          return
        }

        // 调用后端删除 API，假设是 DELETE /records/records/filename/{filename} 或者 DELETE /records/records/带query参数
        // 你的前端代码使用了 params: { filename: selectedFilename.value }，这通常对应于 DELETE /records/records/?filename=...
        // 确保后端路由匹配
        const { data } = await axios.delete(
          'http://127.0.0.1:8000/api/records/', // 确保 API 路径正确
          {
            headers: { Authorization: `Bearer ${token}` },
            params: {
              // 通过 params 传递作为查询参数
              filename: selectedFilename.value
            }
          }
        )

        ElMessage.success(data.detail || '记录删除成功！')

        // 删除成功后，更新前端数据：从原始 results 数组中移除相关记录
        // 注意：这里的过滤逻辑取决于后端是删除所有记录还是只删除与 filename 匹配的记录
        // 假设后端是删除所有与 filename 匹配的记录
        results.value = results.value.filter(
          (item) => item.filename !== selectedFilename.value
        )

        // watch groupedResults 会自动处理 selectedFilename 的更新逻辑（跳到下一项或清空）
        // 如果删除的是当前选中的文件，watch 会将 selectedFilename 更新为新的第一个文件 (如果还有其他文件)
      } catch (error) {
        console.error('Error deleting record:', error)
        ElMessage.error(
          '删除记录失败: ' + (error.response?.data?.detail || error.message)
        )
      }
    })
    .catch(() => {
      // 用户取消删除操作
      ElMessage.info('已取消删除')
    })
}

// Helper function to get category color class for left border
const getCategoryColorClass = (category) => {
  // 根据你的后端颜色映射定义前端 CSS 类名
  const colorMap = {
    个人隐私信息: 'red',
    黄色信息: 'yellow',
    不良有害信息: 'blue',
    无敏感信息: 'green'
    // "分类失败": "gray" // 如果需要且后端可能返回
  }
  // 如果类别不在映射中，或者 category 为 null/undefined/空字符串，使用默认样式
  return colorMap[category] || 'default'
}

// Helper function to get ElTag type
const getCategoryTagType = (category) => {
  const typeMap = {
    个人隐私信息: 'danger', // 红色标签
    黄色信息: 'warning', // 黄色标签
    不良有害信息: 'danger', // 蓝色标签 (Element Plus没有蓝色标签，用 danger 突出不良信息)
    无敏感信息: 'success' // 绿色标签
    // "分类失败": "info" // 灰色标签 (Element Plus没有灰色标签)
  }
  // 根据你的后端分类结果映射到 Element Plus 的标签类型
  return typeMap[category] || 'info' // 默认为 info 类型标签
}

// -----------------------------

onMounted(() => {
  fetchRecords() // 组件挂载时加载记录
})
</script>

<style scoped>
/* 容器样式 */
.result-container {
  max-width: 900px; /* 增加最大宽度 */
  margin: 20px auto; /* 居中并调整外边距 */
  padding: 20px; /* 调整内边距 */
  background-color: #fff; /* 白色背景 */
  border-radius: 8px; /* 圆角 */
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); /* 增加阴影 */
}

/* 下拉框样式 */
.select-image {
  width: 350px; /* 适当加宽以显示文件名 */
  margin-bottom: 20px; /* 底部外边距 */
}

/* 结果卡片样式 */
.result-card {
  margin-bottom: 20px; /* 底部外边距 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); /* 卡片阴影 */
  border-radius: 8px; /* 卡片圆角 */
}

/* 卡片头部样式 */
.card-header {
  display: flex;
  justify-content: space-between; /* 标题和按钮分布在两端 */
  align-items: center;
  font-size: 18px; /* 调整标题字体大小 */
  font-weight: bold; /* 标题加粗 */
  color: #333; /* 标题颜色 */
  padding-bottom: 10px; /* 底部内边距 */
  border-bottom: 1px solid #eee; /* 底部细线 */
}

/* 图片预览区域样式 */
.preview-img {
  max-width: 100%; /* 最大宽度 */
  height: auto; /* 高度自适应 */
  display: block; /* 块级元素 */
  margin: 10px auto 20px auto; /* 居中并调整上下外边距 */
  border: 1px solid #ddd; /* 边框 */
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05); /* 阴影 */
  border-radius: 4px; /* 圆角 */
}

/* 图片加载失败/加载中的占位符样式 */
.image-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 300px; /* 占位高度 */
  background: var(--el-fill-color-light);
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
.image-slot .el-icon {
  margin-right: 5px;
  font-size: 24px;
}
.image-slot.is-loading .el-icon {
  animation: rotating 2s linear infinite; /* 加载图标旋转动画 */
}
@keyframes rotating {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* 颜色图例样式 */
.color-legend {
  margin-top: 15px; /* 调整上边距 */
  margin-bottom: 20px; /* 底部外边距 */
  padding: 15px; /* 内边距 */
  background-color: #f0f9eb; /* Element Plus Success Light 3 */
  border-radius: 8px; /* 圆角 */
  border: 1px solid #e1f3d8; /* Element Plus Success Light 7 */
  color: #67c23a; /* Element Plus Success */
  font-size: 0.9em; /* 字体大小 */
}

.color-legend ul {
  list-style: none;
  padding: 0;
  margin: 10px 0 0 0;
  display: flex; /* 使列表项水平排列 */
  flex-wrap: wrap; /* 允许换行 */
  gap: 20px; /* 增加列表项之间的间距 */
}

.color-legend li {
  display: flex;
  align-items: center;
}

.color-swatch {
  display: inline-block;
  width: 18px; /* 色块大小 */
  height: 18px;
  margin-right: 10px; /* 色块与文字间距 */
  border-radius: 4px; /* 圆角 */
  border: 1px solid #ccc; /* 边框 */
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); /* 阴影 */
}

/* 定义色块颜色样式 */
.color-swatch.red {
  background-color: red;
}
.color-swatch.yellow {
  background-color: yellow;
}
.color-swatch.blue {
  background-color: blue;
}
.color-swatch.green {
  background-color: green;
}
.color-swatch.gray {
  background-color: gray;
}

/* 检测详情列表容器 */
.detections-list {
  margin-top: 15px;
}

/* 每个检测项的样式 */
.detection-item {
  margin-bottom: 15px; /* 底部外边距 */
  padding: 15px; /* 内边距 */
  border-radius: 6px; /* 圆角 */
  border: 1px solid #e0e0e0; /* 默认边框 */
  transition: all 0.3s ease; /* 添加过渡动画 */
  background-color: #fff; /* 白色背景 */
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05); /* 阴影 */
  word-break: break-word; /* 保证长文本换行 */
}

/* 检测项悬停效果 */
.detection-item:hover {
  border-color: #a0cfff; /* 悬停时边框颜色变化 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); /* 悬停时阴影变化 */
}

/* 动态添加的类别颜色左边框 */
.detection-item.category-red {
  border-left: 4px solid red;
}
.detection-item.category-yellow {
  border-left: 4px solid yellow;
}
.detection-item.category-blue {
  border-left: 4px solid blue;
}
.detection-item.category-green {
  border-left: 4px solid green;
}
.detection-item.category-gray {
  border-left: 4px solid gray;
}

/* 文本、分类、置信度行的样式 */
.text-result,
.sensitive-result,
.confidence {
  font-size: 15px; /* 调整字体大小 */
  margin: 8px 0; /* 调整上下外边距 */
  color: #555; /* 文字颜色 */
}
.text-result strong,
.sensitive-result strong,
.confidence strong {
  color: #333; /* 加粗文字颜色 */
  margin-right: 5px; /* 与内容间隔 */
}

/* 敏感信息分类结果样式 */
.sensitive-result .el-tag {
  margin-left: 0; /* 标签与冒号对齐 */
  font-size: 13px; /* 标签字体大小 */
  height: auto; /* 高度自适应内容 */
  line-height: 1.2; /* 行高 */
  padding: 4px 8px; /* 内边距 */
}

/* 置信度分数样式 */
.confidence-score {
  font-weight: bold;
  color: #409eff;
}

/* 标题样式 */
.section-title {
  font-size: 16px;
  font-weight: bold;
  color: #555;
  margin-top: 0;
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px dashed #ddd;
}

/* 骨架屏样式 */
.el-skeleton {
  padding: 20px;
}
</style>
