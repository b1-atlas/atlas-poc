import axios from 'axios';
import moment from 'moment';
import {
  HOME_FETCH_FORECAST_PRICE_LIST_BEGIN,
  HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS,
  HOME_FETCH_FORECAST_PRICE_LIST_FAILURE,
  HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR,
} from './constants';

const dateFormat = 'YYYYMMDD';

function parseData(data) {
  return Object.entries(data).map(d => {
    // const date = [d[0].slice(0, 4), d[0].slice(4, 6), d[0].slice(6, 8)].join('-');
    return {
      date: moment(d[0], dateFormat).toDate(),
      price: +d[1].Price || 0,
      quantity: +d[1].Quantity || 0,
    };
  });
}

// Rekit uses redux-thunk for async actions by default: https://github.com/gaearon/redux-thunk
// If you prefer redux-saga, you can use rekit-plugin-redux-saga: https://github.com/supnate/rekit-plugin-redux-saga
export function fetchForecastPriceList(args = {}, dataRange = 'all') {
  console.log('dataRange', dataRange);
  return dispatch => {
    // optionally you can have getState as the second argument
    dispatch({
      type: HOME_FETCH_FORECAST_PRICE_LIST_BEGIN,
    });

    // Return a promise so that you could control UI flow without states in the store.
    // For example: after submit a form, you need to redirect the page to another when succeeds or show some errors message if fails.
    // It's hard to use state to manage it, but returning a promise allows you to easily achieve it.
    // e.g.: handleSubmit() { this.props.actions.submitForm(data).then(()=> {}).catch(() => {}); }
    const promise = new Promise((resolve, reject) => {
      // doRequest is a placeholder Promise. You should replace it with your own logic.
      // See the real-word example at:  https://github.com/supnate/rekit/blob/master/src/features/home/redux/fetchRedditReactjsList.js
      // args.error here is only for test coverage purpose.
      // FROM: '20160401',
      // TO: '20160731',
      const doRequest = axios.get('http://10.58.137.250:5050/forecast/', {
        headers: {
          DATARANGE: dataRange,
        },
      });
      doRequest.then(
        res => {
          dispatch({
            type: HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS,
            data: parseData(res.data),
          });
          resolve(res);
        },
        // Use rejectHandler as the second argument so that render errors won't be caught.
        err => {
          dispatch({
            type: HOME_FETCH_FORECAST_PRICE_LIST_FAILURE,
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
export function dismissFetchForecastPriceListError() {
  return {
    type: HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR,
  };
}

export function reducer(state, action) {
  switch (action.type) {
    case HOME_FETCH_FORECAST_PRICE_LIST_BEGIN:
      // Just after a request is sent
      return {
        ...state,
        fetchForecastPriceListPending: true,
        fetchForecastPriceListError: null,
      };

    case HOME_FETCH_FORECAST_PRICE_LIST_SUCCESS:
      // The request is success
      return {
        ...state,
        fetchForecastPriceListPending: false,
        fetchForecastPriceListError: null,
        forecastPriceList: action.data,
      };

    case HOME_FETCH_FORECAST_PRICE_LIST_FAILURE:
      // The request is failed
      return {
        ...state,
        fetchForecastPriceListPending: false,
        fetchForecastPriceListError: action.data.error,
      };

    case HOME_FETCH_FORECAST_PRICE_LIST_DISMISS_ERROR:
      // Dismiss the request failure error
      return {
        ...state,
        fetchForecastPriceListError: null,
      };

    default:
      return state;
  }
}
