<template>
  <div class="user-center-container">
    <!-- 个人信息展示卡片 -->
    <el-card class="info-card">
      <template #header>
        <div class="card-title">
          个人信息
          <!-- 编辑按钮 -->
          <el-button
            type="primary"
            size="small"
            style="float: right"
            @click="dialogVisible = true"
          >
            修改信息
          </el-button>
        </div>
      </template>
      <div class="info-item">
        <span class="label">用户名：</span
        ><span class="value">{{ userInfo.username }}</span>
      </div>
      <div class="info-item">
        <span class="label">邮箱：</span
        ><span class="value">{{ userInfo.email || '未绑定' }}</span>
      </div>
      <div class="info-item">
        <span class="label">手机号：</span
        ><span class="value">{{ userInfo.phone || '未绑定' }}</span>
      </div>
    </el-card>

    <!-- 修改密码卡片 -->
    <el-card class="password-card">
      <template #header>
        <div class="card-title">修改密码</div>
      </template>
      <el-form
        :model="passwordForm"
        ref="passwordFormRef"
        label-width="100px"
        :rules="passwordRules"
      >
        <el-form-item label="旧密码" prop="oldPassword">
          <el-input
            v-model="passwordForm.oldPassword"
            type="password"
            placeholder="请输入旧密码"
          />
        </el-form-item>
        <el-form-item label="新密码" prop="newPassword">
          <el-input
            v-model="passwordForm.newPassword"
            type="password"
            placeholder="请输入新密码"
          />
        </el-form-item>
        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input
            v-model="passwordForm.confirmPassword"
            type="password"
            placeholder="请确认新密码"
          />
        </el-form-item>
        <el-form-item style="text-align: right; margin-top: 20px">
          <el-button type="primary" @click="handlePasswordChange"
            >提交修改</el-button
          >
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 弹窗：修改信息 -->
    <el-dialog title="修改信息" v-model="dialogVisible" width="400px">
      <el-form :model="editForm" label-width="80px">
        <el-form-item label="用户名">
          <el-input v-model="editForm.username" />
        </el-form-item>
        <el-form-item label="邮箱">
          <el-input v-model="editForm.email" />
        </el-form-item>
        <el-form-item label="手机号">
          <el-input v-model="editForm.phone" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitUpdate">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import axios from 'axios'

const router = useRouter()

// 用户信息
const userInfo = reactive({
  username: '',
  email: '',
  phone: ''
})

// 修改信息弹窗
const dialogVisible = ref(false)
const editForm = reactive({
  username: '',
  email: '',
  phone: ''
})

// 弹窗打开时初始化表单数据
watch(dialogVisible, (visible) => {
  if (visible) {
    Object.assign(editForm, userInfo)
  }
})

// 修改密码表单
const passwordForm = reactive({
  oldPassword: '',
  newPassword: '',
  confirmPassword: ''
})

// 表单校验规则
const passwordRules = {
  oldPassword: [{ required: true, message: '请输入旧密码', trigger: 'blur' }],
  newPassword: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, max: 20, message: '密码长度需在6-20位之间', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认新密码', trigger: 'blur' },
    {
      validator: (rule, value, callback) => {
        if (value !== passwordForm.newPassword) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
}

// 获取当前用户信息
const fetchUserInfo = async () => {
  try {
    const res = await axios.get('http://localhost:8000/api/user/me', {
      headers: {
        Authorization: `Bearer ${localStorage.getItem('access_token')}`
      }
    })
    Object.assign(userInfo, res.data)
  } catch (error) {
    ElMessage.error('获取用户信息失败，请重新登录')
    // router.push('/login')
  }
}

// 提交密码修改请求
const handlePasswordChange = async () => {
  try {
    await axios.post(
      'http://localhost:8000/api/user/change-password',
      {
        old_password: passwordForm.oldPassword,
        new_password: passwordForm.newPassword,
        confirm_password: passwordForm.confirmPassword
      },
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`,
          'Content-Type': 'application/json'
        }
      }
    )
    ElMessage.success('密码修改成功，请重新登录')
    localStorage.removeItem('access_token')
    router.push('/login')
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '密码修改失败')
  }
}

// 提交修改用户信息请求
const submitUpdate = async () => {
  try {
    await axios.put(
      'http://localhost:8000/api/user/update-info',
      {
        username: editForm.username,
        email: editForm.email,
        phone: editForm.phone
      },
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access_token')}`
        }
      }
    )
    ElMessage.success('信息修改成功')
    dialogVisible.value = false
    fetchUserInfo()
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || '信息修改失败')
  }
}

// 初始加载
onMounted(fetchUserInfo)
</script>

<style scoped>
.user-center-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.info-card,
.password-card {
  margin-bottom: 30px;
  padding: 30px;
}

.info-item {
  display: flex;
  align-items: center;
  padding: 10px 0;
  font-size: 16px;
}

.label {
  width: 80px;
  color: #666;
  font-weight: 500;
}

.value {
  color: #333;
  margin-left: 10px;
}

.card-title {
  font-size: 20px;
  font-weight: 600;
  color: #2c3e50;
}
</style>
