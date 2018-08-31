import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import moment from 'moment';
// import { Link } from 'react-router-dom';
import * as actions from './redux/actions';
import {
  Layout,
  Breadcrumb,
  Spin,
  Alert,
  Form,
  Row,
  Col,
  Button,
  Select,
  DatePicker,
  Tabs,
  Slider,
  InputNumber,
  Checkbox,
} from 'antd';
import InteractiveChart from './InteractiveChart';
import BarChart from './BarChart';
const { Header, Content, Footer } = Layout;
const FormItem = Form.Item;
const Option = Select.Option;
const { RangePicker } = DatePicker;
const CheckboxGroup = Checkbox.Group;
const TabPane = Tabs.TabPane;
const dateFormat = 'YYYY-MM-DD';
const startDate = '2011-01-01';
const endDate = '2017-03-31';

export class DefaultPage extends Component {
  static propTypes = {
    home: PropTypes.object.isRequired,
    actions: PropTypes.object.isRequired,
  };

  state = {
    initializing: true,
    forecasting: false,
    startDate: startDate,
    endDate: endDate,
    dataRangeType: 'all',
    recommending: false,
    weekdays: ['0', '1', '2', '3', '4', '5', '6'],
    confidence: 0.5,
    period: 7,
  };

  componentWillMount() {
    this.props.actions
      .fetchPurchasePriceList()
      .then(() => {
        this.setState({
          initializing: false,
        });
      })
      .catch(() => {
        this.setState({
          initializing: false,
        });
      });
  }

  handleDateChange = (date, dateString) => {
    // console.log(date, dateString);
    this.props.actions.cleanForecastData();
    if (dateString[0] !== '') {
      this.setState({
        startDate: dateString[0],
        endDate: dateString[1],
      });
      this.setState({
        dataRangeType: 'custom',
      });
    } else {
      this.setState({
        startDate,
        endDate,
        dataRangeType: 'all',
      });
    }
  };

  disabledDate = current => {
    // Can not select days before today and today
    return (
      current &&
      (current.toDate() < moment(startDate, dateFormat).toDate() ||
        current.toDate() > moment(endDate, dateFormat).toDate())
    );
  };

  handleForecast = evt => {
    evt.preventDefault();
    this.setState({
      forecasting: true,
    });
    const dataRange = this.state.dataRangeType != 'year' ? 'all' : 'year';
    this.props.actions
      .fetchForecastPriceList({}, dataRange)
      .then(() => {
        this.setState({
          forecasting: false,
        });
      })
      .catch(() => {
        this.setState({
          forecasting: false,
        });
      });
  };

  handleRecommend = evt => {
    evt.preventDefault();
    this.setState({
      recommending: true,
    });
    this.props.actions
      .fetchRecommendation({}, this.state.period, this.state.weekdays, this.state.confidence)
      .then(() => {
        this.setState({
          recommending: false,
        });
      })
      .catch(() => {
        this.setState({
          recommending: false,
        });
      });
    // setTimeout(() => {
    //   this.setState({
    //     recommending: false,
    //   });
    // }, 1000);
  };

  handleDateLink = evt => {
    // console.log(evt.target);
    const className = evt.target.className;
    if (className === this.state.dataRangeType) {
      evt.preventDefault();
      return false;
    }

    this.props.actions.cleanForecastData();
    if (className.includes('all')) {
      this.setState({
        startDate,
        endDate,
        dataRangeType: 'all',
      });
    } else if (className.includes('year')) {
      this.setState({
        startDate: '2016-03-31',
        endDate: '2017-03-31',
        dataRangeType: 'year',
      });
    }
  };

  handlePeriodChange = value => {
    this.setState({
      period: value,
    });
    // console.log('checked = ', value);
  };

  handleWeekdaysChange = checkedValues => {
    this.setState({
      weekdays: checkedValues,
    });
    // console.log('checked = ', checkedValues);
  };

  handleConfidenceChange = value => {
    this.setState({
      confidence: value,
    });
  };

  renderInitializing() {
    return (
      <div className="home-default-page">
        <Layout>
          <Header style={{ position: 'fixed', zIndex: 1, width: '100%' }}>
            <div className="app-title">
              <h2>Atlas POC</h2>
            </div>
          </Header>
          <Content style={{ padding: '0 50px', marginTop: 64 }}>
            <Breadcrumb style={{ margin: '16px 0', textAlign: 'left' }}>
              <Breadcrumb.Item>Hackathon</Breadcrumb.Item>
              <Breadcrumb.Item>App</Breadcrumb.Item>
            </Breadcrumb>
            <div
              className="app-conent"
              style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 188px)' }}
            >
              <Spin />
            </div>
          </Content>
          <Footer style={{ textAlign: 'center' }}>Atlas team @The B1 Dev China Hackathon </Footer>
        </Layout>
      </div>
    );
  }

  renderError(err) {
    return (
      <Alert
        message={
          typeof err === 'string'
            ? err
            : 'Fetch data failed, please check your network or try again later.'
        }
        type="error"
        closable
        showIcon
        onClose={this.dismissFetchListError}
      />
    );
  }

  render() {
    const {
      fetchPurchasePriceListPending,
      fetchPurchasePriceListError,
      purchasePriceList,
      fetchForecastPriceListError,
      forecastPriceList,
      fetchRecommendationError,
      recommendationList,
    } = this.props.home;
    console.log('recommendationList', recommendationList);
    const weekdaysOptions = [
      { label: 'Mon', value: '1' },
      { label: 'Tue', value: '2' },
      { label: 'Wed', value: '3' },
      { label: 'Thu', value: '4' },
      { label: 'Fri', value: '5' },
      { label: 'Sat', value: '6' },
      { label: 'Sun', value: '0' },
    ];

    if (fetchPurchasePriceListPending || this.state.initializing) {
      return this.renderInitializing();
    }

    return (
      <div className="home-default-page">
        <Layout>
          <Header style={{ position: 'fixed', zIndex: 1, width: '100%' }}>
            <div className="app-title">
              <h2>Atlas POC</h2>
            </div>
          </Header>
          <Content style={{ padding: '0 50px', marginTop: 64 }}>
            <Breadcrumb style={{ margin: '16px 0', textAlign: 'left' }}>
              <Breadcrumb.Item>Hackathon</Breadcrumb.Item>
              <Breadcrumb.Item>Atlas App</Breadcrumb.Item>
            </Breadcrumb>
            <div
              className="app-content"
              style={{ background: '#fff', padding: 24, minHeight: 'calc(100vh - 188px)' }}
            >
              <fieldset>
                <legend>Configuration</legend>
                <Form className="ant-advanced-search-form" onSubmit={this.handleForecast}>
                  <Row gutter={24}>
                    <Col span={20}>
                      <FormItem label="Past Data Range:">
                        <RangePicker
                          onChange={this.handleDateChange}
                          defaultValue={[
                            moment(startDate, dateFormat),
                            moment(endDate, dateFormat),
                          ]}
                          value={[
                            moment(this.state.startDate, dateFormat),
                            moment(this.state.endDate, dateFormat),
                          ]}
                          format={dateFormat}
                          disabledDate={this.disabledDate}
                          dateRender={current => {
                            const style = {};
                            // console.log(moment(, dateFormat).toDate());
                            if (
                              current.format(dateFormat) === startDate ||
                              current.format(dateFormat) === endDate
                            ) {
                              style.border = '1px solid #1890ff';
                              style.borderRadius = '50%';
                            }
                            return (
                              <div className="ant-calendar-date" style={style}>
                                {current.date()}
                              </div>
                            );
                          }}
                        />
                        <span className="date-links">
                          <a
                            className={
                              this.state.dataRangeType === 'all' ? 'all link active' : 'all link'
                            }
                            onClick={this.handleDateLink}
                          >
                            All Past Data
                          </a>
                          <a
                            className={
                              this.state.dataRangeType === 'year' ? 'year link active' : 'year link'
                            }
                            onClick={this.handleDateLink}
                          >
                            Last Year Data
                          </a>
                        </span>
                      </FormItem>
                    </Col>
                    <Col span={4}>
                      <Button
                        size="large"
                        type="primary"
                        htmlType="submit"
                        loading={this.state.forecasting}
                      >
                        {this.state.forecasting ? 'Forecast...' : 'Forecast'}
                      </Button>
                    </Col>
                  </Row>
                  <Row className="form-bottom-buttons-container">
                    <Col span="24" style={{ textAlign: 'right' }} />
                  </Row>
                </Form>
              </fieldset>
              <fieldset style={{ margin: '10px 0 0 0' }}>
                <legend>Intelligent Forecast and Recommendation</legend>
                <Tabs defaultActiveKey="price">
                  <TabPane tab="Purchase Price Forecast" key="price">
                    <div className="chart-container price">
                      {fetchPurchasePriceListError && this.renderError(fetchPurchasePriceListError)}
                      {fetchForecastPriceListError && this.renderError(fetchForecastPriceListError)}
                      <div className="chart-legend">
                        <span className="line past" />
                        <label>Past Data</label>
                        <span className="line forecast" />
                        <label>Forecast Data</label>
                      </div>
                      <InteractiveChart
                        id="price-chart"
                        yAxisText="Price (CNY)"
                        measure="price"
                        unit="¥"
                        data={purchasePriceList}
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        forecastedData={forecastPriceList}
                        forecasting={this.state.forecasting}
                      />
                      {this.state.forecasting && <Spin />}
                    </div>
                  </TabPane>
                  <TabPane tab="Sales Quantity Forecast" key="quantity">
                    {fetchPurchasePriceListError && this.renderError(fetchPurchasePriceListError)}
                    {fetchForecastPriceListError && this.renderError(fetchForecastPriceListError)}
                    <div className="chart-container quantity">
                      <div className="chart-legend">
                        <span className="line past" />
                        <label>Past Data</label>
                        <span className="line forecast" />
                        <label>Forecast Data</label>
                      </div>
                      <InteractiveChart
                        id="quantity-chart"
                        yAxisText="Quantity (pcs)"
                        measure="quantity"
                        unit=""
                        data={purchasePriceList}
                        startDate={this.state.startDate}
                        endDate={this.state.endDate}
                        forecastedData={forecastPriceList}
                        forecasting={this.state.forecasting}
                      />
                      {this.state.forecasting && <Spin />}
                    </div>
                  </TabPane>
                  <TabPane tab="Purchase Decisions Recommendation" key="recommend">
                    <Form className="ant-advanced-search-form" onSubmit={this.handleRecommend}>
                      <Row gutter={24}>
                        <Col span={8}>
                          <FormItem label="Inventory Period:">
                            <Select defaultValue="7" onChange={this.handlePeriodChange}>
                              <Option value="1">1 Day</Option>
                              <Option value="7">7 Days</Option>
                              <Option value="15">15 Days</Option>
                              <Option value="30">1 Month</Option>
                              <Option value="c">Customize</Option>
                            </Select>
                          </FormItem>
                        </Col>
                        <Col span={12}>
                          <FormItem label="Purchase Days:">
                            <CheckboxGroup
                              options={weekdaysOptions}
                              defaultValue={['0', '1', '2', '3', '4', '5', '6']}
                              onChange={this.handleWeekdaysChange}
                            />
                          </FormItem>
                        </Col>
                        <Col span={4}>
                          <Button
                            size="large"
                            type="primary"
                            htmlType="submit"
                            loading={this.state.recommending}
                          >
                            {this.state.recommending ? 'Recommend...' : 'Recommend'}
                          </Button>
                        </Col>
                      </Row>
                      <Row gutter={24}>
                        <Col>
                          <FormItem label="Purchase Frequency:">
                            <Row>
                              <Col span={10}>
                                <Slider
                                  min={0}
                                  max={1}
                                  onChange={this.handleConfidenceChange}
                                  value={this.state.confidence}
                                  step={0.1}
                                />
                              </Col>
                              <Col span={4}>
                                <InputNumber
                                  min={0}
                                  max={1}
                                  style={{ marginLeft: 16 }}
                                  step={0.1}
                                  value={this.state.confidence}
                                  onChange={this.handleConfidenceChange}
                                />
                              </Col>
                            </Row>
                          </FormItem>
                        </Col>
                      </Row>
                    </Form>
                    {fetchRecommendationError && this.renderError(fetchRecommendationError)}
                    <br />
                    {recommendationList.length > 0 && (
                      <div className="chart-container recommend">
                        <strong>
                          Here are the purchase quantity recommendation of each days for next 3
                          months:
                        </strong>
                        <div className="chart-legend">
                          <span className="line recommend" />
                          <label>Recommendation Data</label>
                        </div>
                        <Row gutter={24}>
                          <Col span={12}>
                            {' '}
                            <BarChart
                              height={350}
                              title="Purchase Decisions for April"
                              data={recommendationList.filter(d => d.x.indexOf('04-') >= 0)}
                            />
                          </Col>
                          <Col span={12}>
                            <BarChart
                              height={350}
                              title="Purchase Decisions for May"
                              data={recommendationList.filter(d => d.x.indexOf('05-') >= 0)}
                            />
                          </Col>
                        </Row>
                        <Row gutter={24}>
                          <Col span={12}>
                            {' '}
                            <BarChart
                              height={350}
                              title="Purchase Decisions for June"
                              data={recommendationList.filter(d => d.x.indexOf('06-') >= 0)}
                            />
                          </Col>
                        </Row>
                      </div>
                    )}
                  </TabPane>
                </Tabs>
              </fieldset>
            </div>
          </Content>
          <Footer style={{ textAlign: 'center' }}>Atlas team #B1 Hackathon 2018# </Footer>
        </Layout>
      </div>
    );
  }
}

/* istanbul ignore next */
function mapStateToProps(state) {
  return {
    home: state.home,
  };
}

/* istanbul ignore next */
function mapDispatchToProps(dispatch) {
  return {
    actions: bindActionCreators({ ...actions }, dispatch),
  };
}

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(DefaultPage);
