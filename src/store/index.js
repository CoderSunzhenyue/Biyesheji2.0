import { createStore } from 'vuex'

export default createStore({
  state: {
    user: null
  },
  mutations: {
    SET_USER(state, user) {
      state.user = user
    }
  },
  actions: {
    login({ commit }, credentials) {
      // 这里应该调用API，这里简化处理
      commit('SET_USER', { username: credentials.username })
    },
    logout({ commit }) {
      commit('SET_USER', null)
    }
  },
  getters: {
    isAuthenticated: (state) => !!state.user
  }
})
