import axios from 'axios';
import moment from 'moment';
import {
  HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN,
  HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS,
  HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE,
  HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR,
} from './constants';

const dateFormat = 'YYYYMMDD';

function parseData(data) {
  return Object.entries(data).map(d => {
    return {
      date: moment(d[0], dateFormat).toDate(),
      price: +d[1].Price || 0,
      quantity: +d[1].Quantity || 0,
    };
  });
}
// Rekit uses redux-thunk for async actions by default: https://github.com/gaearon/redux-thunk
// If you prefer redux-saga, you can use rekit-plugin-redux-saga: https://github.com/supnate/rekit-plugin-redux-saga
export function fetchPurchasePriceList(args = {}) {
  return dispatch => {
    // optionally you can have getState as the second argument
    dispatch({
      type: HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN,
    });

    // Return a promise so that you could control UI flow without states in the store.
    // For example: after submit a form, you need to redirect the page to another when succeeds or show some errors message if fails.
    // It's hard to use state to manage it, but returning a promise allows you to easily achieve it.
    // e.g.: handleSubmit() { this.props.actions.submitForm(data).then(()=> {}).catch(() => {}); }
    const promise = new Promise((resolve, reject) => {
      // doRequest is a placeholder Promise. You should replace it with your own logic.
      // See the real-word example at:  https://github.com/supnate/rekit/blob/master/src/features/home/redux/fetchRedditReactjsList.js
      // args.error here is only for test coverage purpose.
      // http://10.58.137.250:5050/getdata/ http://localhost:6075/getdata
      const doRequest = axios.get('http://10.58.137.250:5050/getdata/', {
        // headers: { FROM: '20160331', TO: '20170331' },
      });
      doRequest.then(
        res => {
          dispatch({
            type: HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS,
            data: parseData(res.data),
          });
          resolve(res);
        },
        // Use rejectHandler as the second argument so that render errors won't be caught.
        err => {
          dispatch({
            type: HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE,
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
export function dismissFetchPurchasePriceListError() {
  return {
    type: HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR,
  };
}

export function reducer(state, action) {
  switch (action.type) {
    case HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN:
      // Just after a request is sent
      return {
        ...state,
        fetchPurchasePriceListPending: true,
        fetchPurchasePriceListError: null,
      };

    case HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS:
      // The request is success
      return {
        ...state,
        fetchPurchasePriceListPending: false,
        fetchPurchasePriceListError: null,
        purchasePriceList: action.data,
      };

    case HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE:
      // The request is failed
      return {
        ...state,
        fetchPurchasePriceListPending: false,
        fetchPurchasePriceListError: action.data.error,
      };

    case HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR:
      // Dismiss the request failure error
      return {
        ...state,
        fetchPurchasePriceListError: null,
      };

    default:
      return state;
  }
}
