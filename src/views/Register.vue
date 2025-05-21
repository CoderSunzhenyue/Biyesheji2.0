<template>
  <div class="register">
    <el-form
      ref="registerRef"
      :model="registerForm"
      :rules="registerRules"
      class="register-form"
    >
      <h3 class="title">图片安全检测系统 - 注册</h3>
      <el-form-item prop="username">
        <el-input
          v-model="registerForm.username"
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
          v-model="registerForm.password"
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
      <el-form-item prop="confirmPassword">
        <el-input
          v-model="registerForm.confirmPassword"
          type="password"
          size="large"
          auto-complete="off"
          placeholder="确认密码"
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
          @click.prevent="handleRegister"
        >
          <span v-if="!loading">立即注册</span>
          <span v-else>注册中...</span>
        </el-button>
        <div style="float: right">
          <router-link class="link-type" to="/login"
            >已有账号？登录</router-link
          >
        </div>
      </el-form-item>
    </el-form>
    <div class="el-register-footer">
      <span>Copyright © 2025 图像敏感信息检测系统</span>
    </div>
  </div>
</template>

<script setup>
import { ref, getCurrentInstance } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import axios from 'axios' // 确保这一行存在且正确
const { proxy } = getCurrentInstance()
const router = useRouter()

const registerForm = ref({
  username: '',
  password: '',
  confirmPassword: ''
})

const registerRules = {
  username: [{ required: true, trigger: 'blur', message: '请输入用户名' }],
  password: [{ required: true, trigger: 'blur', message: '请输入密码' }],
  confirmPassword: [
    { required: true, trigger: 'blur', message: '请再次输入密码' },
    {
      validator: (rule, value, callback) => {
        if (value !== registerForm.value.password) {
          callback(new Error('两次输入的密码不一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
}

const loading = ref(false)

const handleRegister = async () => {
  proxy.$refs.registerRef.validate(async (valid) => {
    if (!valid) return

    loading.value = true
    try {
      await axios.post('http://localhost:8000/api/auth/register', {
        username: registerForm.value.username,
        password: registerForm.value.password
      })

      ElMessage.success('注册成功，请登录')
      router.push('/login')
    } catch (error) {
      console.error(error)
      if (error.response?.status === 400) {
        ElMessage.error('用户名已存在')
      } else {
        ElMessage.error('注册失败，请重试')
      }
    } finally {
      loading.value = false
    }
  })
}
</script>

<style scoped>
.register {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
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

.register-form {
  border-radius: 12px;
  background: #fff;
  width: 400px;
  padding: 25px 25px 5px 25px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
}
.el-register-footer {
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
