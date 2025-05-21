import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Dashboard from '../views/Dashboard.vue'
import Menu1 from '../views/Menu1.vue'
import Menu2 from '../views/Menu2.vue'
import Menu3 from '../views/Menu3.vue'
import UploadDetect from '../views/UploadDetect.vue'
import System from '../views/System.vue'

import System2 from '../views/System2.vue'
import DetectResult from '../views/DetectResult.vue'
import UserCenter from '../views/UserCenter.vue'
import Register from '../views/Register.vue'

const routes = [
  {
    path: '/',
    redirect: '/login'
  },
  {
    path: '/login',
    component: Login
  },
  {
    path: '/register',
    component: Register
  },
  {
    path: '/dashboard',
    component: Dashboard,
    children: [
      {
        path: 'UploadDetect',
        component: UploadDetect
      },
      {
        path: 'DetectResult',
        component: DetectResult
      },

      {
        path: 'UserCenter',
        component: UserCenter
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
