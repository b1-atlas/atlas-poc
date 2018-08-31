import axios from 'axios';
import moment from 'moment';
import {
  HOME_FETCH_RECOMMENDATION_BEGIN,
  HOME_FETCH_RECOMMENDATION_SUCCESS,
  HOME_FETCH_RECOMMENDATION_FAILURE,
  HOME_FETCH_RECOMMENDATION_DISMISS_ERROR,
} from './constants';

const dateFormat = 'YYYYMMDD';

function parseData(data) {
  return Object.entries(data).map(d => {
    const date = moment(d[0], dateFormat);
    // const day = date.day();
    return {
      x: date.format('MM-DD'),
      y: +d[1].Recommend || 0,
    };
  });
}

// Rekit uses redux-thunk for async actions by default: https://github.com/gaearon/redux-thunk
// If you prefer redux-saga, you can use rekit-plugin-redux-saga: https://github.com/supnate/rekit-plugin-redux-saga
export function fetchRecommendation(args = {}, period, weekdays, confidence) {
  // console.log(period, weekdays, confidence);
  console.log(period, weekdays.join(','), confidence);
  return dispatch => {
    // optionally you can have getState as the second argument
    dispatch({
      type: HOME_FETCH_RECOMMENDATION_BEGIN,
    });

    // Return a promise so that you could control UI flow without states in the store.
    // For example: after submit a form, you need to redirect the page to another when succeeds or show some errors message if fails.
    // It's hard to use state to manage it, but returning a promise allows you to easily achieve it.
    // e.g.: handleSubmit() { this.props.actions.submitForm(data).then(()=> {}).catch(() => {}); }
    const promise = new Promise((resolve, reject) => {
      // doRequest is a placeholder Promise. You should replace it with your own logic.
      // See the real-word example at:  https://github.com/supnate/rekit/blob/master/src/features/home/redux/fetchRedditReactjsList.js
      // args.error here is only for test coverage purpose.
      // http://10.58.137.250:5050/recommend/ http://localhost:6075/recommend
      const doRequest = axios.get('http://localhost:6075/recommend', {
        headers: { PERIOD: period, WEEKDAYS: weekdays.join(','), CONFIDENCE: confidence },
      });
      doRequest.then(
        res => {
          dispatch({
            type: HOME_FETCH_RECOMMENDATION_SUCCESS,
            data: parseData(res.data),
          });
          resolve(res);
        },
        // Use rejectHandler as the second argument so that render errors won't be caught.
        err => {
          dispatch({
            type: HOME_FETCH_RECOMMENDATION_FAILURE,
            data: { error: err },
          });
          reject(err);
        },
      );
    });

    return promise;
  };
}

// Async action saves request error by default, this method is used to dismiss the error info.
// If you don't want errors to be saved in Redux store, just ignore this method.
export function dismissFetchRecommendationError() {
  return {
    type: HOME_FETCH_RECOMMENDATION_DISMISS_ERROR,
  };
}

export function reducer(state, action) {
  switch (action.type) {
    case HOME_FETCH_RECOMMENDATION_BEGIN:
      // Just after a request is sent
      return {
        ...state,
        fetchRecommendationPending: true,
        fetchRecommendationError: null,
      };

    case HOME_FETCH_RECOMMENDATION_SUCCESS:
      // The request is success
      return {
        ...state,
        fetchRecommendationPending: false,
        fetchRecommendationError: null,
        recommendationList: action.data,
      };

    case HOME_FETCH_RECOMMENDATION_FAILURE:
      // The request is failed
      return {
        ...state,
        fetchRecommendationPending: false,
        fetchRecommendationError: action.data.error,
      };

    case HOME_FETCH_RECOMMENDATION_DISMISS_ERROR:
      // Dismiss the request failure error
      return {
        ...state,
        fetchRecommendationError: null,
      };

    default:
      return state;
  }
}
