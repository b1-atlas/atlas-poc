import configureMockStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import nock from 'nock';

import {
  HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN,
  HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS,
  HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE,
  HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR,
} from '../../../../src/features/home/redux/constants';

import {
  fetchPurchasePriceList,
  dismissFetchPurchasePriceListError,
  reducer,
} from '../../../../src/features/home/redux/fetchPurchasePriceList';

const middlewares = [thunk];
const mockStore = configureMockStore(middlewares);

describe('home/redux/fetchPurchasePriceList', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  it('dispatches success action when fetchPurchasePriceList succeeds', () => {
    const store = mockStore({});

    return store.dispatch(fetchPurchasePriceList())
      .then(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS);
      });
  });

  it('dispatches failure action when fetchPurchasePriceList fails', () => {
    const store = mockStore({});

    return store.dispatch(fetchPurchasePriceList({ error: true }))
      .catch(() => {
        const actions = store.getActions();
        expect(actions[0]).toHaveProperty('type', HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN);
        expect(actions[1]).toHaveProperty('type', HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE);
        expect(actions[1]).toHaveProperty('data.error', expect.anything());
      });
  });

  it('returns correct action by dismissFetchPurchasePriceListError', () => {
    const expectedAction = {
      type: HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR,
    };
    expect(dismissFetchPurchasePriceListError()).toEqual(expectedAction);
  });

  it('handles action type HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN correctly', () => {
    const prevState = { fetchPurchasePriceListPending: false };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_PURCHASE_PRICE_LIST_BEGIN }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchPurchasePriceListPending).toBe(true);
  });

  it('handles action type HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS correctly', () => {
    const prevState = { fetchPurchasePriceListPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_PURCHASE_PRICE_LIST_SUCCESS, data: {} }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchPurchasePriceListPending).toBe(false);
  });

  it('handles action type HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE correctly', () => {
    const prevState = { fetchPurchasePriceListPending: true };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_PURCHASE_PRICE_LIST_FAILURE, data: { error: new Error('some error') } }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchPurchasePriceListPending).toBe(false);
    expect(state.fetchPurchasePriceListError).toEqual(expect.anything());
  });

  it('handles action type HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR correctly', () => {
    const prevState = { fetchPurchasePriceListError: new Error('some error') };
    const state = reducer(
      prevState,
      { type: HOME_FETCH_PURCHASE_PRICE_LIST_DISMISS_ERROR }
    );
    expect(state).not.toBe(prevState); // should be immutable
    expect(state.fetchPurchasePriceListError).toBe(null);
  });
});

