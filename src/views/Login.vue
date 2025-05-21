<template>
  <div class="login">
    <el-form
      ref="loginRef"
      :model="loginForm"
      :rules="loginRules"
      class="login-form"
    >
      <h3 class="title">图像敏感信息检测系统</h3>
      <el-form-item prop="username">
        <el-input
          v-model="loginForm.username"
          type="text"
          size="large"
          auto-complete="off"
          placeholder="用户名"
        >
          <template #prefix>
            <el-icon><User /></el-icon>
          </template>
        </el-input>
      </el-form-item>
      <el-form-item prop="password">
        <el-input
          v-model="loginForm.password"
          type="password"
          size="large"
          auto-complete="off"
          placeholder="密码"
          show-password
        >
          <template #prefix>
            <el-icon><Lock /></el-icon>
          </template>
        </el-input>
      </el-form-item>
      <el-form-item style="width: 100%">
        <el-button
          :loading="loading"
          size="large"
          type="primary"
          style="width: 100%"
          @click.prevent="handleLogin"
        >
          <span v-if="!loading">立即登录</span>
          <span v-else>登录中...</span>
        </el-button>
        <div style="float: right">
          <router-link class="link-type" to="/register">立即注册</router-link>
        </div>
      </el-form-item>
    </el-form>
    <div class="el-login-footer">
      <span>Copyright © 2025 图像敏感信息检测系统</span>
    </div>
  </div>
</template>

<script setup>
import { ref, getCurrentInstance } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import axios from 'axios'

const { proxy } = getCurrentInstance()
const router = useRouter()

const loginForm = ref({
  username: '',
  password: ''
})

const loginRules = {
  username: [{ required: true, trigger: 'blur', message: '请输入用户名' }],
  password: [{ required: true, trigger: 'blur', message: '请输入密码' }]
}

const loading = ref(false)

//  handleLogin 方法
const handleLogin = async () => {
  proxy.$refs.loginRef.validate(async (valid) => {
    if (!valid) return ElMessage.error('请完整填写登录信息')

    loading.value = true
    try {
      // 发送 urlencoded 格式数据
      const formData = new URLSearchParams({
        username: loginForm.value.username,
        password: loginForm.value.password
      })

      const response = await axios.post(
        'http://localhost:8000/api/auth/login',
        formData, // 直接传递 URLSearchParams
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded' // 明确格式
          }
        }
      )

      localStorage.setItem('access_token', response.data.access_token)
      router.push('/dashboard/UploadDetect')
      ElMessage.success('登录成功')
    } catch (error) {
      console.error('登录错误：', error.response?.data || error.message)
      if (error.response?.status === 500) {
        ElMessage.error('服务器内部错误，请联系管理员')
      } else {
        ElMessage.error('登录失败，请检查用户名或密码')
      }
    } finally {
      loading.value = false
    }
  })
}
</script>
<style scoped>
.login {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  /* 重要：请根据您在项目中放置图片的位置调整这里的 URL 路径 */
  background-image: url('@/assets/beijing.jpeg');
  background-size: cover; /* 确保图片覆盖整个背景区域 */
  background-position: center; /* 将图片居中 */
  background-repeat: no-repeat; /* 防止图片重复平铺 */
}
.title {
  margin: 0 auto 30px auto;
  text-align: center;
  color: #707070;
}

.login-form {
  border-radius: 12px;
  background: #fff;
  width: 400px;
  padding: 25px 25px 5px 25px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
}
.el-login-footer {
  height: 40px;
  line-height: 40px;
  position: fixed;
  bottom: 0;
  width: 100%;
  text-align: center;
  color: #fff;
  font-size: 12px;
}
.link-type {
  margin-top: 10px;
  display: inline-block;
  color: #409eff;
}
</style>
