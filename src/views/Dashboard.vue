<template>
  <el-container class="dashboard-container">
    <!-- 侧边栏 -->
    <el-aside
      :width="isCollapsed ? '64px' : '200px'"
      class="sidebar"
      :class="{ 'is-collapsed': isCollapsed }"
    >
      <div class="sidebar-header">
        <span class="logo" v-show="!isCollapsed">图像敏感信息检测系统</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapsed"
        mode="vertical"
        background-color="#2c3e50"
        text-color="#fff"
        active-text-color="#3498db"
      >
        <el-menu-item
          index="/dashboard/UploadDetect"
          @click="handleMenuClick('/dashboard/UploadDetect')"
        >
          <el-icon><Picture /></el-icon>
          <template #title>图像检测</template>
        </el-menu-item>

        <el-menu-item
          index="/dashboard/DetectResult"
          @click="handleMenuClick('/dashboard/DetectResult')"
        >
          <el-icon><Document /></el-icon>
          <template #title>检测记录</template>
        </el-menu-item>
        <el-menu-item
          index="/dashboard/UserCenter"
          @click="handleMenuClick('/dashboard/UserCenter')"
        >
          <el-icon><User /></el-icon>
          <template #title>个人中心</template>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 顶部 + 主体 -->
    <el-container>
      <el-header class="dashboard-header">
        <div class="hamburger" @click="isCollapsed = !isCollapsed">
          <el-icon><Menu /></el-icon>
        </div>
        <h3 class="header-title">图像敏感信息检测系统</h3>
        <div class="header-actions">
          <el-button type="text" @click="handleLogout">退出登录</el-button>
        </div>
      </el-header>

      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { Menu, Picture, Document, Setting, User } from '@element-plus/icons-vue'

const isCollapsed = ref(false)
const activeMenu = ref('/dashboard/menu1')
const router = useRouter()

const handleLogout = () => {
  localStorage.removeItem('access_token')
  router.push('/login')
}

const handleMenuClick = (path) => {
  activeMenu.value = path
  router.push(path)
}
</script>

<style scoped>
.dashboard-container {
  min-height: 100vh;
  display: flex;
  flex-direction: row;
}

.sidebar {
  transition: width 0.3s ease;
  background-color: #2c3e50;
}

.sidebar-header {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-bottom: 1px solid #364a63;
}

.logo {
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  background-color: #fff;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  height: 60px;
}

.header-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 500;
}

.header-actions {
  display: flex;
  gap: 16px;
}

.el-main {
  padding: 20px;
  background-color: #f0f2f5;
  flex-grow: 1;
  overflow-y: auto;
}

/* 折叠模式样式优化 */
.el-aside.is-collapsed .el-menu-item .el-menu-item__title {
  display: none;
}

.el-aside.is-collapsed .el-menu-item {
  justify-content: center;
}

/* 响应式 */
@media (max-width: 768px) {
  .dashboard-container {
    flex-direction: column;
  }
  .sidebar {
    width: 100% !important;
    height: 60px;
    display: flex;
    flex-direction: row;
    border-bottom: 1px solid #364a63;
  }
  .sidebar-header {
    flex-grow: 1;
  }
  .el-menu {
    flex-grow: 1;
    display: flex;
    flex-direction: row;
    justify-content: center;
  }
  .el-menu-item {
    margin: 0 16px;
  }
}
</style>
